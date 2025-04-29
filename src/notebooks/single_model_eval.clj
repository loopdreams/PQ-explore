;; # Evaluating a Single Model Config
(ns notebooks.single-model-eval
  (:require
   [clojure.edn :as edn]
   [clojure.java.io :as io]
   [clojure.set :as set]
   [clojure.string :as str]
   [notebooks.generation :as gen]
   [notebooks.llm-api :as llm]
   [notebooks.preparation :refer [ds]]
   [notebooks.rag-evaluation
    :refer [add-all-generation-evaluation-metrics average-coll build-responses-eval-ds-avgs]]
   [notebooks.tokenizer :as tokenizer]
   [scicloj.kindly.v4.kind :as kind]
   [scicloj.tableplot.v1.plotly :as plotly]
   [selmer.parser :as templates]
   [tablecloth.api :as tc]))

;; ## Goal
;; In the previous section we looked at a broad range of models to try to get a
;; sense of the best performing ones in this context.
;;
;; However, we might have been "putting the cart before the horse" a little in
;; that case. It might be better to get the best retrieval/generation 'pipeline'
;; in place first, and then we could swap in different models to test their
;; effectivness.
;;
;; When it comes to RAG applications, there are many different ways you can vary
;; a configuration, and it's hard to know where to start. In this section we'll
;; see if we can narrow our focus in on evaluating:
;;
;;  a. our prompt
;;
;;  b. our retrieval strategy
;;

;; ## LLM-Generated Evaluation Dataset
;;
;; Before we move on to answering our question, it might be better to have a
;; larger evaluation dataset. In the last section, we used a 10-question
;; evaluation dataset that I filled in manually. Ideally, we would have more
;; questions to get a better sense of how a model performs.  Luckily, we can
;; also use an LLM to help us generate some evaluation questions for testing.
;;
;; To generate a questions/answer pair, we will pass the model some context, In
;; this case, we will use full answers from the original dataset, and ask it to
;; generate a single question for each answer it is passed.
;;
;; As a side note, from this exercise I learned that some more cleaning would be
;; needed on the original data, as there are quite a few answers that don't
;; contain useful information, for example, when they simply indicate that a
;; question has been directed elsewhere.

(defn generate-evaluation-question [ctx model-ref]
  (let [prompt (-> "prompts/qa_generation_prompt.txt"
                   slurp
                   (templates/render {:context ctx}))
        response (llm/ask-llm {:model-ref model-ref
                               :question prompt})
        question (re-find #"(?<=question: ).*(?=\n)" response)
        answer (re-find #"(?<=answer: ).*" response)]
    {question [answer]}))

(defn generate-eval-dataset [docs model-ref]
  (reduce merge (mapv #(generate-evaluation-question % model-ref) docs)))

(defn take-n-random-answers [n]
  (let [answers (-> ds
                    (tc/drop-missing :answer)
                    (tc/drop-rows #(re-find #"propose to take" (:answer %)))
                    :answer)]
    (take n (repeatedly #(rand-nth answers)))))

(comment
  (let [qas (-> (take-n-random-answers 20)
                (generate-eval-dataset "gpt-4o-mini"))]
    (spit "data/evaluation_questions/questions_ai_gen_3.edn"
          (with-out-str (clojure.pprint/pprint qas)))))

;; I ran the question generation a few times and combined the AI generated
;; questions in a single file, removing questions that didn't seem reasonable.

(def evaluation-dataset-ai (edn/read-string (slurp "data/evaluation_questions/questions_ai_gen.edn")))

(count evaluation-dataset-ai)

(take 5 evaluation-dataset-ai)

;; As you can see, the LLM-generated questions/answers are quite specific, and
;; look like a good way to test how well our RAG can find information based on a
;; user's question.
;;
;; Now that we have an evaluation dataset, we can try to build a pipeline to
;; evaluate the performance of a model with different settings applied.

;; ## Evaluation Pipeline
;;
;; Here, I'll chain together some of the functions from previous notebooks to build a
;; single 'pipeline' for evaluating a model.
;;
;; The aim will be to test 4 different generation approaches:

;; 1. Prompt A with chunked answers database (**A1**)
;;
;; 2. Prompt A with questions database (**A2**)
;;
;; 3. Prompt B with chunked answers database (**B1**)
;;
;; 4. Prompt B with questions database (**B2**)
;;
;; In other words we are trying to vary the prompts, and the retrieval
;; strategy.
;;
;; As a reminder, in earlier sections we explored different retrieval
;; approaches. One involved splitting up the 'answer' column in the original
;; dataset into 'chunks', and then storing these in the database. We saw that
;; the potential optimum 'chunk size' was around 3 sentences per chunk.
;;
;; The alternative, initial approach was to store questions from the original
;; dataset in a vector database, retrieve similar questions to the user's
;; question, and then use these to 'look up' their answers in the original
;; dataset. The logic here was that if similar questions have been asked
;; previously, then these answers could be re-used as context for the LLM.
;;
;; The first approach, above, is the standard way of creating a vector database
;; for a RAG, but we can test below to see if it really is the best approach.

;; Our second variable to explore is the prompt.

;; Our initial promot was as follows:
;;
;;> I want you to act as a responsible and trustworthy senior government
;;> official. Please provide an answer to a citizen's question, using only the
;;> context provided. Answer as if you are talking directly to the citizen and be
;;> neutral and formal as possible. If you can't find a specific detail from the
;;> question, please acknowledge this and provide any other helpful information
;;> that may be related to the question.  If you can't find sufficient
;;> information in the context to answer the question at all, then reply with \"I
;;> am unable to answer this question with the information I have available.\"

;; For the second prompt, we'll actually try something relatively similar. The
;; point here will be to see if the results vary dramatically with only small
;; changes to the prompt. I'll mainly drop the word 'please' and also refer to
;; the question-asker as a 'user' as opposed to a 'citizen'.

(defn get-rag-answer-alt-prompt [rag-data db-store-name add-prompt-fn]
  (-> rag-data
      (gen/add-context db-store-name)
      add-prompt-fn
      gen/add-llm-response))

(defn add-alt-prompt [{:keys [retrieved-context] :as rag-data}]
  (->> (str
        "You are a responsible and trustworthy senior government official.
       Provide an answer to the user's question, using only the context
       provided. Answer as if you are talking directly to the user and make sure
       the answer is neutral and formal. If you can't find the specific detail
       that the user is looking for from the question, acknowledge this and
       provide other helpful information from the context that may be related to
       the question. If you can't find any information in the context to answer
       the question, then reply with \"I am unable to answer this question with
       the information I have available.\" "
        "\n\n CONTEXT: " (str/join "\n\n" retrieved-context))
      (assoc rag-data :system-prompt)))

(def prompt-A-fn (partial gen/add-pq-prompt))
(def prompt-B-fn (partial add-alt-prompt))


;; The function below takes the following arguments and adds the LLM answers for each of the test questions.
;;
;; - An evaluation dataset
;;
;; - An LLM model reference
;;
;; - A vector store reference
;;
;; - A prompt-generation function
;;
;; We will focus on varying the last two of these.

(defn get-llm-responses [evaluation-dataset model db-store-name add-prompt-fn]
  (reduce (fn [res [question ground-truth]]
            (let [answer (get-rag-answer-alt-prompt {:question question
                                                     :model-ref model
                                                     :ground-truth ground-truth}
                                                    db-store-name
                                                    add-prompt-fn)]
              (conj res answer)))
          []
          evaluation-dataset))

;; Next, a function to add 'retrieval' metrics, to see if our retrieval method
;; is finding the right documents based on the ground-truth answers in the
;; evaluation dataset. This is repeating what we've covered earlier, but the
;; goal here is to have the full picture in the final dataset.

(defn add-retrieval-metrics-single [rag-data]
  (let [target    (str/join " " (:ground-truth rag-data))
        retrieved (str/join " " (:retrieved-context rag-data))]
    (-> (tokenizer/calculate-retrieval-metrics target retrieved :word)
        (set/rename-keys {:IoU       :retrieval-IoU
                          :precision :retrieval-precision
                          :recall    :retrieval-recall})
        (dissoc :label)
        (merge rag-data))))

(defn add-retrieval-metrics [rag-data]
  (mapv add-retrieval-metrics-single rag-data))

;; Finally, a function to put these all together.

(defn generate-and-evaluate-answers [eval-dataset generation-model vector-db-name prompt-fn evaluation-model]
  (-> eval-dataset
      (get-llm-responses generation-model vector-db-name prompt-fn)
      (add-retrieval-metrics)
      (add-all-generation-evaluation-metrics evaluation-model)))

(defn write-rag-data! [fname data]
  (spit fname
        (with-out-str (clojure.pprint/pprint data))))

(def generation-model "gemini-2.5-flash-preview-04-17")
(def evaluation-model "o4-mini-2025-04-16")

;; I'm using the OpenAI model 'o4-mini' again for evaluation, just to be
;; consistent with the previous notebook.

(defn label-results [results label]
  (reduce (fn [labelled res]
            (conj labelled
                  (assoc res :label label)))
          []
          results))


;; ### Running the tests
(comment
  ;; 1. Prompt A with chunked answers ("A1")
  (time
   (let [results (-> (generate-and-evaluate-answers evaluation-dataset-ai
                                                    generation-model
                                                    :db-docs
                                                    prompt-A-fn
                                                    evaluation-model)
                     (label-results "A1"))
         fname   (str "data/single_model_eval/" generation-model "_1.edn")]
     (write-rag-data! fname results)))
  ;; Elapsed time: 667154.140542 msecs

  ;; 2. Prompt A with question retrieval method ("A2")
  (time
   (let [results (-> (generate-and-evaluate-answers evaluation-dataset-ai
                                                    generation-model
                                                    :question-retrieval
                                                    prompt-A-fn
                                                    evaluation-model)
                     (label-results "A2"))
         fname   (str "data/single_model_eval/" generation-model "_2.edn")]
     (write-rag-data! fname results)))
  ;; "Elapsed time: 821583.118417 msecs"
  ;;
  ;; 3. Prompt B with chunked answers ("B1")
  (time
   (let [results (-> (generate-and-evaluate-answers evaluation-dataset-ai
                                                    generation-model
                                                    :db-docs
                                                    prompt-B-fn
                                                    evaluation-model)
                     (label-results "B1"))
         fname   (str "data/single_model_eval/" generation-model "_3.edn")]
     (write-rag-data! fname results)))
  ;; "Elapsed time: 671342.190417 msecs"
  ;;
  ;; 4. Prompt B with question retrieval method ("B2")
  ;; "Elapsed time: 766714.09675 msecs"
  (time
   (let [results (-> (generate-and-evaluate-answers evaluation-dataset-ai
                                                    generation-model
                                                    :question-retrieval
                                                    prompt-B-fn
                                                    evaluation-model)
                     (label-results "B2"))
         fname   (str "data/single_model_eval/" generation-model "_4.edn")]
     (write-rag-data! fname results))))

;; ### Results Dataset

(def ds-eval-results
  (let [file-data (->> (rest (file-seq (io/file "data/single_model_eval")))
                       (mapv (comp edn/read-string slurp))
                       (reduce into))]
    (-> file-data
        (tc/dataset)
        (tc/map-columns :llm-%-correctness [:metric-llm-correctness-score] #(when % (float (/ % 5))))
        (tc/map-columns :llm-%-relevance [:metric-llm-relevance-score] #(when % (float (/ % 3)))))))

(tc/column-names ds-eval-results)

(tc/row-count ds-eval-results)


;; ## Exploring the results
(defn chart-pivot-data [metrics]
  (-> ds-eval-results
      (tc/group-by [:label])
      (tc/aggregate metrics)
      (tc/pivot->longer (complement #{:label}))
      (tc/rename-columns {:$column :metric
                          :$value :value})))

(defn pivot-chart [data]
  (-> data
      (tc/order-by :label)
      (plotly/base
       {:=width 800})
      (plotly/layer-bar
       {:=x :metric
        :=y :value
        :=color :label})))

;; ### LLM Answer Generation Metrics
;; These are the metrics that were added by our evaluation model (OpenAI
;; 4o-mini) based on the evaluation prompts.

(def llm-metrics-chart-spec
  (chart-pivot-data
   {:faithfulness #(average-coll (:metric-llm-faithfulness-score %))
    :correctness #(average-coll (:llm-%-correctness %))
    :relevance #(average-coll (:llm-%-relevance %))}))

(pivot-chart llm-metrics-chart-spec)

;; In relation to our two scenarios:
;;
;; - The slight changes to the prompt resulted in very slight differences in the
;;   results, which is good to see (there is some predictability there). We can't
;;   see exactly *why* the second prompt was slightly better than the first, but
;;   it a useful result nonetheless. Overall the second prompt (removing the
;;   word 'please' and changing a few of the sentences, but otherwise leaving
;;   the instructions mostly the same) with the document-retrieval method worked
;;   the best, and achieved 100% faithfulness.
;;
;; - A much more significant difference emerges in relation to the best
;;   retrieval strategy. Retrieving the information from the chunked answers
;;   directly seems to do much better than my initial instinct of searching
;;   through similar questions first.
;;
;; Even though we tested retrieval earlier, viewing the potential impact that
;; different retrieval strategies can have on the generated response is very
;; useful.

;; ### Deterministic Answer Generation Metrics
;; These metrics measure the generated responses against the answers in the
;; evaluation dataset, using things like token overlap.

(def token-overlap-gen-metrics-chart-spec
  (chart-pivot-data
   {:recall       #(average-coll (:token-overlap-recall %))
    :precision    #(average-coll (:token-overlap-precision %))
    :f1           #(average-coll (:token-overlap-f1 %))
    :faithfulness #(average-coll (:token-overlap-faithfulness %))}))

(pivot-chart token-overlap-gen-metrics-chart-spec)

(def rouge-gen-metrics-chart-spec
  (chart-pivot-data
   {:recall       #(average-coll (:rouge-l-recall %))
    :precision    #(average-coll (:rouge-l-precision %))
    :f1           #(average-coll (:rouge-l-f1 %))
    :faithfulness #(average-coll (:rouge-faithfulness %))}))

(pivot-chart rouge-gen-metrics-chart-spec)

;; ### Semantic Answer Generation Metrics
;; This metric shows the cosine similarity between the generated answer and the
;; evaluation dataset answer.

(def semantic-overlap-gen-metrics-chart-spec
  (chart-pivot-data
   {:semantic-similarity #(average-coll (:cosine-similarity %))}))

(pivot-chart semantic-overlap-gen-metrics-chart-spec)

;; ### Retrieval Metrics
;; Similar to the 'deterministic' metrics above, these metrics evaluated token
;; overlap beween the retrieved context and the answers in the evaluation
;; dataset.

(def retrieval-metrics-chart-spec
  (chart-pivot-data
   {:precision #(average-coll (:retrieval-precision %))
    :IoU #(average-coll (:retrieval-IoU %))}))

(def retrieval-recall-metrics-chart-spec
  (chart-pivot-data
   {:recall #(average-coll (:retrieval-recall %))}))

(pivot-chart retrieval-metrics-chart-spec)
(pivot-chart retrieval-recall-metrics-chart-spec)


;; ### Retrieval Impact on Generation

;; Let's have a quick look at the relationship between some of the retrieval
;; metrics and the generation metrics.

(defn scatter-plot-comparison [ds retrieval-metric generation-metric]
  (-> ds
      (plotly/base
       {:=x retrieval-metric
        :=y generation-metric})
      (plotly/layer-point
       {:=color :label})))

;; #### Retrieval vs Token Overlap
;; The relationship between retrieval metrics and the token overlap between the
;; generated answers and ground-truth answers.

(-> (scatter-plot-comparison ds-eval-results :retrieval-IoU :token-overlap-f1)
    (plotly/layer-smooth
     {:=name "Predicted"}))

(-> (scatter-plot-comparison ds-eval-results :retrieval-recall :token-overlap-recall)
    (plotly/layer-smooth
     {:=name "Predicted"}))

(-> (scatter-plot-comparison ds-eval-results :retrieval-precision :token-overlap-precision)
    (plotly/layer-smooth
     {:=name "Predicted"}))

;; #### Retrieval vs Semantic Similarity
;; A similar comparison with the semantic similarity metric (more of an exponential relationship)

(scatter-plot-comparison ds-eval-results :retrieval-precision :cosine-similarity)
(scatter-plot-comparison ds-eval-results :retrieval-IoU :cosine-similarity)


;; #### Retrieval vs the LLM-evaluated Metrics

(-> ds-eval-results
    (tc/drop-missing :metric-llm-faithfulness-score)
    (tc/map-columns :faithfull? [ :metric-llm-faithfulness-score]
                    (fn [score]
                      (if (= score 1) "Faithfull" "Not Faithfull")))
    (plotly/layer-boxplot
     {:=y :retrieval-IoU
      :=x :faithfull?}))

(-> ds-eval-results
    (tc/map-columns :correct? [ :metric-llm-correctness-score]
                    (fn [score]
                      (if (> score 3) "Correct" "Poor Correctness Score")))
    (plotly/layer-boxplot
     {:=y :retrieval-IoU
      :=x :correct?}))

(-> ds-eval-results
    (tc/map-columns :relevant? [ :metric-llm-relevance-score]
                    (fn [score]
                      (if (> score 2) "Very Relevant" "Not So Relevant")))
    (plotly/layer-boxplot
     {:=y :retrieval-IoU
      :=x :relevant?}))



;; ### Example Responses

;; Best and worst answers by llm-metrics and semantic similarity

(def results-sort-order [:metric-llm-faithfulness-score
                         :metric-llm-correctness-score
                         :metric-llm-relevance-score
                         :cosine-similarity])
(-> ds-eval-results
    (tc/order-by results-sort-order :desc)
    (tc/select-columns [:question :answer :ground-truth :label])
    (tc/select-rows (range 5)))


(-> ds-eval-results
    (tc/drop-missing :metric-llm-faithfulness-score)
    (tc/order-by results-sort-order)
    (tc/select-columns [:question :answer :label :metric-llm-faithfulness-score
                        :metric-llm-correctness-score
                        :metric-llm-relevance-score])
    (tc/select-rows (range 3)))

;; Hmm, it seems like the 'worst performing' answers were actually honestly
;; answered by the LLM.
;;
;; On the one hand, we would definitely want to score these answers lower,
;; because it is indicating a problem in the RAG chain (at the retrieval leval).
;; However, on the other hand it might be nice to build in recognition that
;; these answers, at least, didn't make up relevant information.


;; Let's filter out these questions and see what the worst performing answers were.

(-> ds-eval-results
    (tc/drop-missing :metric-llm-faithfulness-score)
    (tc/drop-rows #(re-find #"unable to|cannot find the specific|does not contain" (:answer %)))
    (tc/order-by results-sort-order)
    (tc/select-columns [:question :answer :ground-truth :label])
    (tc/select-rows (range 3)))

;; We can see an important potential error here. For the question on the
;; Emergency Flooding Scheme, the question-retrieval method seems to not only
;; fail in returning the relevant information, it seems to mislead the model by
;; returning the *wrong* information. Let's have a look at that context to see.

(-> ds-eval-results
    (tc/select-rows #(= (:label %) "B2"))
    (tc/select-rows #(re-find #"Emergency Humanitarian Flooding scheme" (:question %)))
    :retrieved-context
    first
    (nth 3))

;; This seems to be the context that caused the confusion. After a quick Google
;; search, it seems that DETE is responsible for administering the scheme to
;; **small businesses**, but the correct answer is indeed "The Irish Red Cross"
;; as was indicated in the evaluation dataset.
;;
;; This is yet another example of why lots of testing and fine tuning is needed
;; to perfect this kind of application.

;; Let's finally see how many questions in the dataset were 'unanswerable', due
;; to the retrieval method not providing context. It seems there are two ways to
;; filter for this, (a) the model provided the default answer, and (b) the
;; answer received a low correctness score.


(-> ds-eval-results
    (tc/select-rows #(or (< (:metric-llm-correctness-score %) 2)
                         (re-find #"I am unable to answer this question with the information I have available." (:answer %))))
    (tc/group-by [:label])
    (tc/aggregate {:num tc/row-count}))

;; The question retrieval method did quite badly, failing on around 20-28% of
;; the questions. The chunked-docs retrieval method failed on 3 questions (6%).


;; ## Summary
;;
;; The main conclusion from this section is the importance of the **document
;; retrieval** method used to provide context for the RAG.
;;
;; It seems like this can be improved in a few ways:
;;
;; - More data cleaning and preparation on the data before storing it in a
;;   vector database. For example, doing more work to remove potentially
;;   duplicate/redundant  information.
;;
;; - Experiment and test different strategies for breaking up the information
;;   before storing it.
;;
;; The best approach here would also depend heavily on the end-use case. For
;; example, if you only wanted to retrieve specific pieces of information (as in
;; the evaluation dataset used in this section), then smaller, more precise
;; chunks of information seem to work better. However, if you wanted to answer
;; very broad questions about policy or strategy, then maybe you would want
;; larger chunks of information stored.
;;
;; From what I could see, the model is very capable of finding the relevant
;; information from the context, once the context is properly provided.
;;
;; Overall, some other key takeaways were:
;;
;; - Developing a RAG system means tinkering with multiple different
;;   parts/modules within a pipeline. Clojure's emphasis on immutable,
;;   composable functions seems to make this kind of work very easy and
;;   intuitive.
;;
;; - However, it seems like there is potentially a gap in the clojure ecosystem
;;   in terms of Natural Language Processing techniques/libraries. Or, at least
;;   in the short time to develop this project before the conference, I ended up
;;   relying on work in python for some of the token-based/semantic metrics. I
;;   also had to rely on java for the vector database (although I found this
;;   process relatively straight-forward). I would say that I spent around half
;;   the time for this project figuring out how to 'reproduce' python material
;;   in clojure, and half thinking about the actual questions related to the
;;   data. Which, in my case (a hobbyist programmer) was totally fine - I
;;   learned a lot by doing it!  However, if there was some kind of project
;;   deadline this kind of consideration might tip the balance if favour of just
;;   using python, where there already seems to be so much work put into RAG
;;   evaluation libraries.
;;
;; - I should have spent more time at the outset curating and cleaning the
;;   initial dataset. I would love, for example, to have seen how the models
;;   performed when there was information that was potentially out of date. In
;;   my case, the 'window' for data was very short (2 months), which would not
;;   make sense in an actual application of this type.
;;
;; ## Next Steps
;;
;; Finally, here are a couple of ideas that occurred to me throughout the work,
;; but which I didn't get time to explore:
;;
;; - As we saw above, the LLM-generated metrics seem to be much better at
;;   spotting errors than the other metric types. The creators of the
;;   [continuous-eval
;;   repository](https://github.com/relari-ai/continuous-eval/tree/main) have a
;;   very interesting article on [generation
;;   evaluation.](https://blog.relari.ai/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d)
;;   In it they discuss evaluation in a more 'practical' context, where, for
;;   example, the time and cost of evaluating a pipeline using another LLM would
;;   become a significant factor. So, they construct a 'hybrid' pipline that (a)
;;   uses a combination of the deterministic/semantic metrics to do a first pass
;;   on the data, and then (b) filter out datapoints where the metric cannot
;;   decide with confidence if the answer is acceptable. For these 'low
;;   confidence' evaluations, you would then pass them to a LLM-evaluator model
;;   to clarify. In their test they only had to use the LLM reviews on 7% of
;;   answer, saving 15x on costs.
;;
;; - It would be interesting to apply similar evaluation frameworks to the
;;   original, human-generated answers in the dataset. This would be with the
;;   aim of helping policymakers identify strengths and weaknesses in how
;;   information is provided in this important public setting.
;;
