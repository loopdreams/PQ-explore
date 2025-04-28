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
    :refer [add-all-generation-evaluation-metrics model-performance-summary average-coll build-responses-eval-ds-avgs]]
   [notebooks.tokenizer :as tokenizer]
   [notebooks.vdb-evaluation
    :as vdb-eval
    :refer [db-store-chunked-answers generate-context]]
   [notebooks.vector-database :refer [db-store-questions]]
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
;; evaluation dataset. Ideally, we would have more questions to get a better
;; sense of how a model performs.  Luckily, we can also use an LLM to help us
;; generate some evaluation questions for testing.
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

;; Now that we have an evaluation dataset, we can try to build a pipeline to
;; evaluate the performance of a model with different settings applied.

;; ## Evaluation Pipeline
;;
;; Here, I'll chain together some of the functions from previous notebooks to build a
;; single 'pipeline' for evaluating a model.
;;
;; The aim will be to test 4 different generation approaches:

;; 1. Prompt A with chunked answers database (A1)
;;
;; 2. Prompt A with questions database (A2)
;;
;; 3. Prompt B with chunked answers database (B1)
;;
;; 4. Prompt B with questions database (B2)
;;
;; In other words we are trying to vary the prompts, and the retrieval
;; strategy.
;;
;; As a reminder, in earlier sections we explored different retrieval
;; approaches. One involved spilitting up the 'answer' column in the original
;; dataset into 'chunks', and then storing these in the database. We saw that
;; the potential optimum 'chunk size' was around 3 sentences per chunk.
;;
;; The alternative, initial approach was to store questions from the original
;; dataset in a vector database, retrieve similar questions to the user's
;; question, and then use these to 'look up' their answers in the origianl
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

(defn get-rag-answer-alt-prompt [rag-data db-store add-prompt-fn]
  (-> rag-data
      (gen/add-context db-store)
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
;; - A prompt generation function
;;
;; We will focus on varying the last two of these.

(defn get-llm-responses [evaluation-dataset model db-store add-prompt-fn]
  (reduce (fn [res [question ground-truth]]
            (let [answer (get-rag-answer-alt-prompt {:question question
                                                     :model-ref model
                                                     :ground-truth ground-truth}
                                                    db-store
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

(defn generate-and-evaluate-answers [eval-dataset generation-model vector-db prompt-fn evaluation-model]
  (-> eval-dataset
      (get-llm-responses generation-model vector-db prompt-fn)
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
                                                    vdb-eval/db-store-chunked-answers
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
                                                    db-store-questions
                                                    prompt-A-fn
                                                    evaluation-model)
                     (label-results "A2"))
         fname   (str "data/single_model_eval/" generation-model "_2.edn")]
     (write-rag-data! fname results)))
  ;; "Elapsed time: 785598.29325 msecs"
  ;;
  ;; 3. Prompt B with chunked answers ("B1")
  (time
   (let [results (-> (generate-and-evaluate-answers evaluation-dataset-ai
                                                    generation-model
                                                    vdb-eval/db-store-chunked-answers
                                                    prompt-B-fn
                                                    evaluation-model)
                     (label-results "B1"))
         fname   (str "data/single_model_eval/" generation-model "_3.edn")]
     (write-rag-data! fname results)))
  ;; "Elapsed time: 671342.190417 msecs"
  ;;
  ;; 4. Prompt B with question retrieval method ("B2")
  ;; "Elapsed time: 792633.33325 msecs"
  (time
   (let [results (-> (generate-and-evaluate-answers evaluation-dataset-ai
                                                    generation-model
                                                    db-store-questions
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

;; ### LLM Retrieval Metric
;; These are the metrics that were added by our evaluation model (OpenAI
;; 4o-mini) based on the evaluation prompts.

(def llm-metrics-chart-spec
  (chart-pivot-data
   {:faithfulness #(average-coll (:metric-llm-faithfulness-score %))
    :correctness #(average-coll (:llm-%-correctness %))
    :relevance #(average-coll (:llm-%-relevance %))}))

(pivot-chart llm-metrics-chart-spec)

;; ### Deterministic Retrieval Metrics
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

;; ### Semantic Generation Metrics
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


(def retrieval-gen-comparison-spec
  (-> ds-eval-results
      (tc/group-by [:label])
      (tc/aggregate {:retrieval-recall #(average-coll (:retrieval-recall %))
                     :retrieval-precision #(average-coll (:retrieval-precision %))
                     :retrieval-IoU #(average-coll (:retrieval-IoU %))
                     :generation-faithfulness #(/ (apply + (remove nil? (:metric-llm-faithfulness-score %)))
                                                  (tc/row-count %))
                     :generation-correctness #(/ (average-coll (remove nil? (:metric-llm-correctness-score %))) 5)
                     :generation-relevance #(/ (average-coll (remove nil? (:metric-llm-relevance-score %))) 3)})))


(-> retrieval-gen-comparison-spec
    (tc/order-by :label)
    (plotly/base
     {:=x :label})
    (plotly/layer-bar
     {:=y :retrieval-recall})
    (plotly/layer-bar
     {:=y :generation-faithfulness})
    (plotly/layer-bar
     {:=y :generation-correctness}))


(defn scatter-plot-comparison [ds retrieval-metric generation-metric]
  (-> ds
      (plotly/layer-point
       {:=x retrieval-metric
        :=y generation-metric
        :=color :label})))

;; Let's compare the retrieval metrics to some of the generation metrics to see
;; if there is any clear correlation.

(scatter-plot-comparison ds-eval-results :retrieval-IoU :token-overlap-f1)

(scatter-plot-comparison ds-eval-results :retrieval-recall :token-overlap-recall)

(scatter-plot-comparison ds-eval-results :retrieval-precision :token-overlap-precision)

(scatter-plot-comparison ds-eval-results :retrieval-IoU :cosine-similarity)

(scatter-plot-comparison ds-eval-results :retrieval-IoU :llm-%-similarity)



;; ### Example Responses
;; #### A1 - Prompt A with chunked docs

;; Best and worst answers by llm-metrics
(-> ds-eval-results
    (tc/order-by [:metric-llm-faithfulness-score
                  :metric-llm-correctness-score
                  :metric-llm-relevance-score]
                 :desc)
    (tc/select-columns [:question :answer :label])
    (tc/select-rows (range 3)))

(-> ds-eval-results
    (tc/drop-missing :metric-llm-faithfulness-score)
    (tc/order-by [:metric-llm-faithfulness-score
                  :metric-llm-correctness-score
                  :metric-llm-relevance-score])
    (tc/select-columns [:question :answer :label :metric-llm-faithfulness-score
                        :metric-llm-correctness-score
                        :metric-llm-relevance-score])
    (tc/select-rows (range 3)))

;; Hmm, it seems like the 'worst performing' answers were actually honestly
;; answered by the LLM. You can see below, that the issue was that the retrieval
;; method didn't find the relevant information - they were all using the
;; 'question retrieval' method.
;;
;; On the one hand, we would definitely want to score these answers lower,
;; because it is indicating a problem in the RAG chain (at the retrieval leval).
;; However, on the other hand it might be nice to build in recognition that
;; these answers, at least, didn't make up relevant information.

(-> ds-eval-results
    (tc/select-rows #(= (:answer %)
                        "I am unable to answer this question with the information I have available."))
    (tc/select-columns [:retrieved-context :question :answer :label]))

;; Let's filter out these questions and see what the worst performing answers were.

(-> ds-eval-results
    (tc/drop-missing :metric-llm-faithfulness-score)
    (tc/drop-rows #(= (:answer %) "I am unable to answer this question with the information I have available."))
    (tc/order-by [:metric-llm-faithfulness-score
                  :metric-llm-correctness-score
                  :metric-llm-relevance-score])
    (tc/select-columns [:question :answer :label :metric-llm-faithfulness-score
                        :metric-llm-correctness-score
                        :metric-llm-relevance-score])
    (tc/select-rows (range 3)))

;; Again, these were the result of issues with the retrieval method.
;;
(-> ds-eval-results
    (tc/drop-missing :metric-llm-faithfulness-score)
    (tc/drop-rows #(= (:answer %) "I am unable to answer this question with the information I have available."))
    (tc/select-rows #(or (= (:label %) "A1")
                         (= (:label %) "B1")))
    (tc/order-by [:metric-llm-faithfulness-score
                  :metric-llm-correctness-score
                  :metric-llm-relevance-score])
    (tc/select-columns [:question :answer :label :metric-llm-faithfulness-score
                        :metric-llm-correctness-score
                        :metric-llm-relevance-score])
    (tc/select-rows (range 3)))

(-> ds-eval-results
    (tc/select-rows #(= (:question %) "What amount was paid to recipients of the Fuel Allowance in November 2023?"))
    (tc/select-columns [:question :answer :label :retrieved-context]))

;; It seems like neither of the retrieval strategies could find the relevant
;; information for this question.
;;
;; This answer does indeed exist, you can see the sentence below. I checked this
;; in case the evaluation model had hallucianated the question/answer.  ("They
;; are also in receipt of Fuel Allowance and were awarded the €300 Fuel Bonus
;; and the €400 Cost of Living Bonus on 22 November 2023.")

(-> ds
    (tc/drop-missing :answer)
    (tc/select-rows #(re-find #"fuel allowance" (str/lower-case (:answer %))))
    (tc/select-rows #(re-find  #"November 2023" (:answer %)))
    (tc/select-rows (range 1)))

;; It seems that, although I did attempt to de-duplicate the documents in the
;; vector store, some duplicates still made it through (it appears to be due to
;; an extra 'space' character in one of the answers). When duplicate information
;; is returned, it leaves less space for the potential for the target
;; information to come through (since we only get the 5 most similar documents).
;;
;; To fix this we could do some more cleaning on the documents being added to
;; the database to remove redundant/duplicate information, and also consider
;; increaing the number of documents returned by the vector dabase (from 5 to
;; 10, for example).
;;
;; We can also use these results to see what kinds of information is hard to
;; retrieve. Let's look at some of the worst performing questions.
;;
;; We can see from the first 4 or so questions below, the the main issue was
;; with not being able to retrieve the data. While, overall that is not too
;; bad in terms of the 50 questions, it does indicate potential areas for
;; improvement.

(-> ds-eval-results
    (tc/select-rows #(or (= (:label %) "A1")
                         (= (:label %) "B1")))
    (tc/group-by [:question])
    (tc/aggregate {:avg-correct #(/ (apply + (:metric-llm-correctness-score %)) 2)
                   :avg-relevant #(/ (apply + (:metric-llm-relevance-score %)) 2)
                   :answer #(into [] (:answer %))
                   :ground-truth #(first (:ground-truth %))})
    (tc/order-by [:avg-correct :avg-relevant])
    (tc/select-rows (range 4)))

;; Number of answers where the model provided the defaul answer for not enough information available:

(-> ds-eval-results
    (tc/select-rows #(re-find #"I am unable to answer this question with the information I have available." (:answer %)))
    (tc/group-by [:label])
    (tc/aggregate {:num tc/row-count}))

;; The question-retrieval method failed for 18/50 questions (36%) for the first
;; prompt. The doc-retrieval method only failed for 2 of the 50 questions.
;;
;; However, in some of the cases, as per the prompt, the model still tries to
;; provide whatever helpful information is available. It seems like in this
;; cases the answers are assigned a low 'correctness' score. So, to get a better
;; estimation of how many questions were unanswerable we could look at low
;; correctness scores.

(-> ds-eval-results
    (tc/select-rows #(< (:metric-llm-correctness-score %) 2))
    (tc/group-by [:label])
    (tc/aggregate {:num tc/row-count}))

;; A huge portion of the questions were incorrect for the 'question-retrieval'
;; method (almost half). And it looks like around 6% of the doc-retrieval method
;; questions were also incorrect.

(->>
 (-> ds-eval-results
     (tc/select-rows #(< (:metric-llm-correctness-score %) 2))
     :question
     frequencies)
 (sort-by val)
 (reverse))

;; The interesting situation here is when there is only '1' lo

(-> ds-eval-results
    (tc/select-rows #(= (:question %) "When was the Research and Innovation Bill published?"))
    (tc/select-columns [:retrieved-context :label]))


(-> ds-eval-results
    (tc/map-columns :incorrect-answers []))


;; ## Summary
;;
;; Examples of things that weren't tested:
;; - Best (performance/value) evaluation model?
