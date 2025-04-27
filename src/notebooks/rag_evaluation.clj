;; # Generation Evaluation
(ns notebooks.rag-evaluation
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py..] :as py]
            [notebooks.preparation :refer [ds]]
            [selmer.parser :as templates]
            [scicloj.kindly.v4.kind :as kind]
            [notebooks.generation :as gen]
            [scicloj.tableplot.v1.plotly :as plotly]
            [notebooks.llm-api :as llm]
            [clojure.edn :as edn]
            [clojure.string :as str]
            [tablecloth.api :as tc]
            [clojure.java.io :as io]
            [notebooks.vdb-evaluation :as vdb])
  (:import
   (dev.langchain4j.data.segment TextSegment)
   (dev.langchain4j.store.embedding CosineSimilarity)
   (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)))


;; For this section, I will be relying heavily on the [continuous-eval (python)](https://github.com/relari-ai/continuous-eval)
;; metrics and approach for starting to think about how to evaluate the RAG.
;;
;; That repository also has some great links to articles explaining some of the
;; concepts in more detail.
;;
;; As the creators of the project write, there are several kinds of questions
;; you might want to consider when evaluating answer generation:
;;
;; - Do I have to use GPT-4 or would a smaller model work too?
;;
;; - Should I fine-tune an LLM for my RAG application?
;;
;; - Which prompts minimize hallucination the most?
;;
;; - How sensitive are answers to different prompts?
;;
;; - Is the LLM already good enough if I provide the right contexts, and should I focus on improving Retrieval instead?

;; ([source](https://blog.relari.ai/a-practical-guide-to-rag-evaluation-part-2-generation-c79b1bde0f5d))
;;
;; In this exercise, I will only really look at the question of what llm
;; **model** might work best with the data that I have and the prompt/retrieval
;; framework we have already set up.
;;
;; We will focus on three categories of metrics:
;;
;; - Deterministic
;;
;; - Semantic
;;
;; - LLM-based
;;
;; Deterministic metrics are similar to how we measured the retrieval
;; performace; they simply measure the *token overlap* between answers generated
;; by the LLM and some kind of reference/ground-truth answers.
;;
;; The semantic metric is similar to the method of retrieving information from
;; the vector database; it checks how similar two pieces of text are based on
;; vector embeddings.
;;
;; LLM-based metrics utilise another LLM to assign a score to the output. For
;; example, to determine 'answer-correctness', we will ask an LLM to assign a
;; score between 1-5 to a generated answer, based on reference answers that we
;; provide ourselves.
;;
;; ## Evaluation Dataset
;;
;; Before going into the metrics further, we will first create a testing dataset
;; that contains some questions and ground truth answers. I've used 10 fairly
;; random questions based on some of the material in the starting dataset of
;; questions and answers.
;;
;; Ideally, we would use a much larger and more thoughtfully curated evaluation
;; dataset, perhaps with input from domain experts across different question areas.
;; The goal here, however, is simply to test out some evaluation workflows in
;; clojure, so a basic evaluation dataset will have to do for now.
;;
;; Below, we just load that dataset. The 'questions.edn' file is set up as a clojure map,
;; where the questions are keys and the ground truth answers and values.


(def evaluation-dataset
  (let [data         (edn/read-string (slurp "data/evaluation_questions/questions.edn"))
        questions    (keys data)
        ground-truth (vals data)]
    (mapv (fn [question truth] (-> {}
                                   (assoc :question question)
                                   (assoc :ground-truth truth)))
          questions
          ground-truth)))

(kind/table evaluation-dataset)

;; ## Generate LLM Answers
;;
;; Next, we will write a helper function to save llm responses and generate some
;; responses by different llm models. These are the responses that we will
;; evaluate later. As you can see below, I tested 20 different models. Some were
;; locally running small models (around 8B parameters max), and some were
;; cloud-based models from Google, OpenAI and Anthropic.

(defn ask-llm-save-responses! [model questions]
  (let [responses (reduce (fn [res question]
                            (conj res (gen/get-rag-answer (assoc question :model-ref model)
                                                          vdb/db-store-chunked-answers)))
                          [] questions)
        f-name (str "data/responses/" model "_responses.edn")]
    (spit f-name responses)))


(comment
  (ask-llm-save-responses! "gemini-2.0-flash-lite" evaluation-dataset)
  (ask-llm-save-responses! "llama3.1" evaluation-dataset)
  (ask-llm-save-responses! "gpt-3.5-turbo" evaluation-dataset)
  (ask-llm-save-responses! "gemma3:1b" evaluation-dataset)
  (ask-llm-save-responses! "gpt-4o-mini" evaluation-dataset)
  (ask-llm-save-responses! "gpt-4o" evaluation-dataset)
  (ask-llm-save-responses! "o4-mini-2025-04-16" evaluation-dataset)
  (ask-llm-save-responses! "o3-mini" evaluation-dataset)
  (ask-llm-save-responses! "gemini-2.0-flash" evaluation-dataset)
  (ask-llm-save-responses! "claude-3-7-sonnet-20250219" evaluation-dataset)
  (ask-llm-save-responses! "claude-3-5-haiku-20241022" evaluation-dataset)
  (ask-llm-save-responses! "claude-3-haiku-20240307" evaluation-dataset)
  (ask-llm-save-responses! "llama3.2" evaluation-dataset)
  (ask-llm-save-responses! "mistral" evaluation-dataset)
  (ask-llm-save-responses! "llava" evaluation-dataset)
  (ask-llm-save-responses! "deepseek-r1" evaluation-dataset)
  (ask-llm-save-responses! "gemma3:4b" evaluation-dataset) ;x
  (ask-llm-save-responses! "granite3.2" evaluation-dataset)
  (ask-llm-save-responses! "gemini-2.5-pro-preview-03-25" evaluation-dataset) ;x
  (ask-llm-save-responses! "gemini-2.5-flash-preview-04-17" evaluation-dataset))

(defonce responses-ds
  (let [responses-dir "data/responses"
        responses (->> responses-dir
                       (io/file)
                       file-seq
                       rest
                       (map (comp edn/read-string slurp))
                       (reduce into))]
    (tc/dataset responses)))

(tc/row-count responses-ds)

;; Each model answered the 10 questions from the evaluation dataset, so that's
;; 200 responses overall.

;; ## Continuous Eval Metrics Functions
;;
;; Below, I am just creating a wrapper for the Continuous-eval deterministic
;; metrics, and re-writing the LLM metrics in clojure, using the
;; [prompt templates that are provided in the continuous-eval repo](https://github.com/relari-ai/continuous-eval/tree/main/continuous_eval/metrics/generation/text/prompts)
;;
;; For demonstrating how the metrics work, we will use a couple of the generated
;; responses as samples.
;;
;; For the question "How many households were in receipt of HAP payments in
;; 2023?", the data available states that 57,617 households were in receipt of
;; payments at the end of **Q3 2023**. In other words, the full data for 2023
;; was not available at that time. Most of the models seemed to be able to pick
;; up that detail, but one of the lower-powered ones, gemma3(1 billion parameter
;; model) didn't qualify the figure to state that it was only for Q3.
;;
;; Also, the question "Are there plans to further reduce public transport fares?"
;; should be a simple 'no', based on the available data, but the gemma3:1b model
;; also gets this one wrong.

(def sample-gen-responses
  (-> responses-ds
      (tc/select-rows #(and (or (= (:model-ref %) "llama3.1")
                                (= (:model-ref %) "gemma3:1b"))
                            (or (re-find #"receipt of HAP payments" (:question %))
                                (re-find #"transport fares" (:question %)))))))

(-> sample-gen-responses
    (tc/select-columns [:model-ref :question :answer])
    (kind/table))



;; ### Deterministic Metrics

(require-python '[continuous_eval.metrics.generation.text.deterministic :as det])

(defn add-deterministic-metrics [{:keys [answer retrieved-context ground-truth] :as rag-data}]
  (let [faithfullness-spec  {:answer            answer
                             :retrieved_context retrieved-context}
        correctness-spec    {:answer               answer
                             :ground_truth_answers (if (seq ground-truth) ground-truth retrieved-context)}
        faithfulness-scores (into {} (py.. (det/DeterministicFaithfulness) (**compute faithfullness-spec)))
        correctness-scores  (into {} (py.. (det/DeterministicAnswerCorrectness) (**compute correctness-spec)))
        reading-scores      (into {} (py.. (det/FleschKincaidReadability) (compute answer)))]
    (->
     (merge
      faithfulness-scores
      correctness-scores
      reading-scores
      rag-data)
     (clojure.set/rename-keys
      {"flesch_reading_ease"         :flesch-reading-ease
       "flesch_kincaid_grade_level"  :flesch-kincaid-grade-level
       "rouge_l_recall"              :rouge-l-recall
       "rouge_faithfulness"          :rouge-faithfulness
       "rouge_l_precision"           :rouge-l-precision
       "rouge_l_f1"                  :rouge-l-f1
       "rouge_p_by_sentence"         :rouge-p-by-sentence
       "bleu_score_by_sentence"      :bleu-score-by-sentence
       "bleu_faithfulness"           :bleu-faithfulness
       "bleu_score"                  :bleu-score
       "token_overlap_p_by_sentence" :token-overlap-p-by-sentence
       "token_overlap_f1"            :token-overlap-f1
       "token_overlap_precision"     :token-overlap-precision
       "token_overlap_recall"        :token-overlap-recall
       "token_overlap_faithfulness"  :token-overlap-faithfulness}))))

;; Example score for the sample responses:

(-> (mapv add-deterministic-metrics (tc/rows sample-gen-responses :as-maps))
    (tc/dataset)
    (tc/select-columns [:model-ref :question :answer :rouge-l-f1 :token-overlap-f1 :bleu-score])
    (kind/table))

;; The 'F1' scores are the combination of 'precision' and 'recall' metrics. As
;; we saw in previous sections, precision is how much of the generated answer is
;; reflected in the ground truth (i.e., what % of the generated answer is not
;; 'superfluous'), and recall is how much of the ground truth is reflected in
;; the generated answer. The F1 score is the harmonic mean of both these scores,
;; with a score closer to 1 being better. The 'BLEU' score is also better when
;; it is closer to 1.
;;
;; In this case, even though these metrics don't check for semantic meaning or
;; logic, the metrics do indicate that the llama3.1 responses were slightly
;; better than the gemma3 responses.

;; ### Semantic Similarity
;;
;; We'll also check, very roughly, the semantic similarity (based on cosine
;; similarity) between the generated responses and the ground truth.

(defn calculate-cosine-similarity [text-a vec-text-b]
  (let [embedding-model (AllMiniLmL6V2EmbeddingModel/new)
        embedding-fn (fn [text]
                       (->> (TextSegment/from text)
                            (. embedding-model embed)
                            (.content)))]
    (CosineSimilarity/between (embedding-fn text-a) (embedding-fn (str/join " " vec-text-b)))))

(defn add-semantic-similarity [{:keys [answer ground-truth] :as rag-data}]
  (let [similarity-score (calculate-cosine-similarity answer ground-truth)]
    (-> rag-data
        (assoc :cosine-similarity similarity-score))))


(add-semantic-similarity {:answer "Paris is the capital of France."
                          :ground-truth ["Berlin in the capital of France."]})

(add-semantic-similarity {:answer "Paris is the capital of France."
                          :ground-truth ["The capital of France is Paris."]})

(add-semantic-similarity {:answer "Paris is the capital of France."
                          :ground-truth ["The capital of France is Paris."
                                         "The Mona Lisa is in Paris."]})

(-> (mapv add-semantic-similarity (tc/rows sample-gen-responses :as-maps))
    (tc/dataset)
    (tc/select-columns [:model-ref :question :answer :cosine-similarity])
    (kind/table))

;; We can see a limitation with this metric here - even though the last
;; gemma3:1b answer is factually inccorrect, it still is quite 'semantically
;; similar' to the ground truth answer.


;; ### LLM Metrics
;;
;; As I mentioned above, I'm using the same approach/prompts for the LLM-based
;; metrics as is used in the continuous-eval project.
;;
;; For example, the 'faithfulness' prompt can be seen below:

(-> "prompts/faithfulness_sys.txt"
    slurp
    (str/replace #"\n" "\n\n")
    (gen/quoted-response)
    kind/md)


(defn add-llm-metric-correctness-score [{:keys [question answer ground-truth] :as rag-data} llm-model]
  (let [system-prompt (slurp "prompts/ans_correctness_sys.txt")
        user-prompt   (-> "prompts/ans_correctness_user.txt"
                          slurp
                          (templates/render {:question     question
                                             :answer       answer
                                             :ground-truth (if (seq ground-truth)
                                                             ground-truth
                                                             (:retrieved-context rag-data))}))
        response      (llm/ask-llm
                       {:model-ref     llm-model
                        :question      user-prompt
                        :system-prompt system-prompt})
        score (first (re-find #"(?<=[S|s]core(.{1,4}))[1|2|3|4|5]" response))
        score (when score (parse-long score))]
    (-> rag-data
        (assoc :metric-llm-correctness-explanation response)
        (assoc :metric-llm-correctness-score score))))

(defn add-llm-metric-faithfulness-score [{:keys [answer retrieved-context] :as rag-data} llm-model]
  (let [system-prompt  (slurp "prompts/faithfulness_sys.txt")
        ret-ctx-joined (str/join "\n" retrieved-context)
        user-prompt    (-> "prompts/faithfulness_user.txt"
                           slurp
                           (templates/render {:answer                   answer
                                              :retrieved-context-joined ret-ctx-joined}))
        response       (llm/ask-llm
                        {:model-ref     llm-model
                         :question      user-prompt
                         :system-prompt system-prompt})
        score (first (re-find #"(?<=[S|s]core(.{1,4}))[yes|no]" (str/lower-case response)))
        score (when score (if (= score "y") 1 0))]
    (-> rag-data
        (assoc :metric-llm-faithfulness-explanation response)
        (assoc :metric-llm-faithfulness-score score))))

(defn add-llm-metric-relevance-score [{:keys [answer question] :as rag-data} llm-model]
  (let [system-prompt  (slurp "prompts/ans_relevance_sys.txt")
        user-prompt    (-> "prompts/ans_relevance_user.txt"
                           slurp
                           (templates/render {:answer answer
                                              :question question}))
        response       (llm/ask-llm
                        {:model-ref     llm-model
                         :question      user-prompt
                         :system-prompt system-prompt})
        score (first (re-find #"(?<=[S|s]core(.{1,4}))[1|2|3]" response))
        score (when score (parse-long score))]
    (-> rag-data
        (assoc :metric-llm-relevance-explanation response)
        (assoc :metric-llm-relevance-score score))))

(defn add-llm-metrics [rag-data model]
  (-> rag-data
      (add-llm-metric-correctness-score model)
      (add-llm-metric-faithfulness-score model)
      (add-llm-metric-relevance-score model)
      (assoc :evaluator-model model)))

;; Finally, let's wrap all of the above three metric types (deterministic,
;; semantic, and llm-based) into a single function.

(defn add-all-generation-evaluation-metrics [responses evaluation-model]
  (mapv (fn [resp]
          (-> resp
              add-deterministic-metrics
              add-semantic-similarity
              (add-llm-metrics evaluation-model)))
        responses))

;; Now, let's use these metrics to evaluate the two example question/answers we genreated earlier.

(comment
  (let [eval-model "gpt-4o"
        output-fname "data/evaluation_example/example.edn"
        sample-with-metrics (add-all-evaluation-metrics
                             (tc/rows sample-gen-responses :as-maps)
                             eval-model)]
    (spit output-fname sample-with-metrics)))

(def sample-gen-responses-metrics (edn/read-string (slurp "data/evaluation_example/example.edn")))

(first sample-gen-responses-metrics)

;; Example LLM Faithfulness evaluation (score can be '1 - faithfull' or '0 - not faithfull'):
(-> sample-gen-responses-metrics
    (tc/dataset)
    (tc/select-columns [:model-ref :question :answer :metric-llm-faithfulness-score :metric-llm-faithfulness-explanation])
    (kind/table))

;; As we can see, the evaluation model correctly identified the errors in the
;; gemma3:1b answers.

;; Example LLM Correctness evaluation (range between 1 and 5):

(-> sample-gen-responses-metrics
    (tc/dataset)
    (tc/select-columns [:model-ref :question :answer :metric-llm-correctness-score :metric-llm-correctness-explanation])
    (kind/table))

;; Example LLM Relevance evaluation (range between 1 and 3):

(-> sample-gen-responses-metrics
    (tc/dataset)
    (tc/select-columns [:model-ref :question :answer :metric-llm-relevance-score :metric-llm-relevance-explanation])
    (kind/table))

;; Interestingly, even though the gemma3 responses were factually incorrect,
;; they still received a high 'relevance' score from the evaluator model. In
;; other words, it recognises that it was still attempting to answer the
;; question in a 'relevant' manner, even though it got the facts wrong.

;; ### Running/Saving evaluations

(defn run-and-save-evaluation-metrics! [responses model]
  (let [model-ref (:model-ref (first responses))
        f-name (str "data/responses_evaluation/" model-ref "_evaluation.edn")
        resp (add-all-generation-evaluation-metrics responses model)]
    (spit f-name resp)))

(defn run-and-save-all-evals! [responses-dir model]
  (let [responses (->> (io/file responses-dir)
                       file-seq
                       rest
                       (mapv (comp edn/read-string slurp)))]
    (mapv #(run-and-save-evaluation-metrics! % model) responses)))

(comment
  ;; 43:55 (very roughly) to run around 15 models
  ;; cost - around 1.44 USD for 18 models * 10 questions each - 180 evaluations
  (run-and-save-all-evals! "data/responses" "o4-mini-2025-04-16"))

;; ## Exploring Performance

;; In this part we'll try to compare the 20 models based on their performance
;; across the metrics.
;;
;; We'll start be defining a few helper functions.

(defn average-coll [coll]
  (float
   (/ (apply + (remove nil? coll))
      (count (remove nil? coll)))))

(defn average-all-cols [numerical-ds]
  (let [cols (tc/column-names numerical-ds)]
    (tc/dataset
     (reduce (fn [res col]
               (assoc res col (average-coll (numerical-ds col))))
             {} cols))))

(defn summarise-model-performance-avgs [rag-datas]
  (let [model-ref (:model-ref (first rag-datas))]
    (-> rag-datas
        (tc/dataset)
        (tc/drop-columns #(re-find #"by-sentence" (name %)))
        (tc/select-columns :type/numerical)
        average-all-cols
        (tc/add-column :model-ref model-ref))))

(defn build-responses-eval-ds-avgs [responses-eval-dir]
  (let [responses (->> responses-eval-dir
                       io/file
                       file-seq
                       rest
                       (mapv (comp edn/read-string slurp))
                       (mapv summarise-model-performance-avgs))]
    (apply tc/concat responses)))

(def ds-performance-averages (build-responses-eval-ds-avgs "data/responses_evaluation"))

(kind/table ds-performance-averages)

(defn concat-responses-eval-data [responses-eval-dir]
  (let [responses (->> responses-eval-dir
                       io/file
                       file-seq
                       rest
                       (mapv (comp edn/read-string slurp)))]
    (reduce into responses)))

(defn add-model-platform [ds]
  (-> ds
      (tc/map-columns :platform [:model-ref]
                      (fn [m]
                        (->
                         (filter #(= (:model-ref %) m) llm/llm-models)
                         first
                         :platform)))))

(defn concat-responses-eval-ds-narrowed [responses-eval-dir]
  (let [ds (tc/dataset (concat-responses-eval-data responses-eval-dir))]
    (-> ds
        (tc/select-columns
         (concat
          (tc/column-names ds :type/numerical)
          [:model-ref :question])))))


(def responses-eval-data (concat-responses-eval-data "data/responses_evaluation"))
(def ds-responses-eval-narrowed (concat-responses-eval-ds-narrowed "data/responses_evaluation"))


(defn make-boxplot [metric]
  (->
   ds-responses-eval-narrowed
   add-model-platform
   (tc/order-by :model-ref)
   (plotly/base
    {:=width 800
     :=color :platform})
   (plotly/layer-boxplot
    {:=x :model-ref
     :=y metric})))

;; ### Deterministic Metrics (non-llm)
;; #### Reading Ease
;;
;; The `flesch-kincaid-grade-level` and `flesch-reading-ease` metrics help show
;; how readable the response is. A lower grade level and higher reading ease
;; level makes the text more readable.
;;

(make-boxplot :flesch-reading-ease)

(make-boxplot :flesch-kincaid-grade-level)


;; Example of max/min reading ease answers

(-> responses-eval-data
    (tc/dataset)
    (tc/select-columns [:flesch-reading-ease :answer])
    (tc/order-by :flesch-reading-ease)
    (tc/select-rows (range 1)))

(-> responses-eval-data
    (tc/dataset)
    (tc/select-columns [:flesch-reading-ease :answer])
    (tc/order-by :flesch-reading-ease :desc)
    (tc/select-rows (range 1)))

;; Let's try a high reading-ease answer with more than 100 words...

(-> responses-eval-data
    (tc/dataset)
    (tc/select-columns [:flesch-reading-ease :answer])
    (tc/map-columns :wc [:answer] (fn [ans]
                                    (-> (str/split ans #"\w+")
                                        (count))))
    (tc/select-rows #(> (:wc %) 100))
    (tc/order-by :flesch-reading-ease :desc)
    (tc/select-rows (range 1)))


;; #### Precision

(-> ds-performance-averages
    add-model-platform
    (plotly/base
     {:=width 800
      :=color :platform
      :=x :model-ref})
    (plotly/layer-bar
     {:=y :token-overlap-precision})
    (plotly/layer-bar
     {:=y :rouge-l-precision}))



;; #### Recall

(-> ds-performance-averages
    add-model-platform
    (plotly/base
     {:=width 800
      :=color :platform
      :=x :model-ref})
    (plotly/layer-bar
     {:=y :token-overlap-recall})
    (plotly/layer-bar
     {:=y :rouge-l-recall}))


;; #### Precision/Recall (F1)

(-> ds-performance-averages
    add-model-platform
    (plotly/base
     {:=width 800
      :=color :platform
      :=x :model-ref})
    (plotly/layer-bar
     {:=y :rouge-l-f1}))

(-> ds-performance-averages
    add-model-platform
    (plotly/base
     {:=width 800
      :=color :platform
      :=x :model-ref})
    (plotly/layer-bar
     {:=y :token-overlap-f1}))


;; ### LLM Generated Metrics
;; #### Faithfulness

(defn make-bar-avgs [metric]
  (->
   ds-performance-averages
   add-model-platform
   (tc/order-by metric)
   (plotly/base
    {:=width 800
     :=color :platform})
   (plotly/layer-bar
    {:=x :model-ref
     :=y metric})))

(make-bar-avgs :metric-llm-faithfulness-score)


;; #### Correctness

(make-bar-avgs :metric-llm-correctness-score)

;; #### Relevance

(make-bar-avgs :metric-llm-relevance-score)


;; ### Individual Performances
;;
;; Let's make a simple 'dashboard' type view to try to get a sense of each
;; model's performance at a glance.
;;
;; We'll introduce an 'indicator' marker to show if the model is performing okay
;; for a metric. Perhaps in an actual evaluation system this could be some kind
;; of target threshold that the model should meet. In this case, we'll just use
;; the averages of all the model performances, so that the indicator will simply
;; indicate if the metric is above/below average.

(defn average-score [ds metrics]
  (->>
   (mapv #(ds %) metrics)
   (reduce into)
   (average-coll)))

(defn eval-averages [ds]
  (-> ds
      (tc/aggregate {:faithfulness #(average-score % [:metric-llm-faithfulness-score])
                     :correctness #(average-score % [:metric-llm-correctness-score])
                     :relevance #(average-score % [:metric-llm-relevance-score])
                     :semantic-similarity #(average-score % [:cosine-similarity])
                     :recall #(average-score % [:token-overlap-recall
                                                :rouge-l-recall])
                     :precision #(average-score % [:rouge-l-precision
                                                   :token-overlap-precision])
                     :f1 #(average-score % [:rouge-l-f1
                                            :token-overlap-f1])})
      (tc/rows :as-maps)
      first))

(def eval-averages-all (eval-averages ds-responses-eval-narrowed))

(defn indicator-symbol [colour]
  [:span {:style (str "color: " colour ";")} "&#11044"])
(def indicator-bad (indicator-symbol "red"))
(def indicator-medium (indicator-symbol "yellow"))
(def indicator-good (indicator-symbol "green"))

;; If above target - green
;; If within less than 10% of target - amber
;; If less than 10% target - red
;;
(defn make-indicator-symbol [value target-value]
  (if (> value target-value) indicator-good
      (let [diff (abs (- target-value value))
            diff-percent (float (/ diff target-value))]
        (if (<= diff-percent 0.1)
          indicator-medium
          indicator-bad))))

(defn model-performance-summary [ds model-ref]
  (let [model-per       (filter #(= (:model-ref %) model-ref) ds)
        faithfulness    (count (filter #(= (:metric-llm-faithfulness-score %) 1) model-per))
        total-questions (count model-per)
        {:keys [correctness
                relevance
                semantic-similarity
                recall
                precision
                f1]} (-> model-per tc/dataset eval-averages)]
    [:div
     [:h1 (name model-ref)]
     [:p (str "Scores based on " total-questions " evaluation questions.")]
     [:table {:style "width: 70%;"}
      [:tr
       [:th "Metric"]
       [:th "Score"]
       [:th "Reference Average"]
       [:th "Status"]]
      [:tr
       [:td "Faithfulness"]
       [:td (str faithfulness "/" total-questions)]
       [:td (:faithfulness eval-averages-all)]
       [:td (make-indicator-symbol (/ faithfulness total-questions) (:faithfulness eval-averages-all))]]
      [:tr
       [:td "Correctness"]
       [:td correctness]
       [:td (:correctness eval-averages-all)]
       [:td (make-indicator-symbol correctness
                                   (:correctness eval-averages-all))]]
      [:tr
       [:td "Relevance"]
       [:td relevance]
       [:td (:relevance eval-averages-all)]
       [:td (make-indicator-symbol relevance
                                   (:relevance eval-averages-all))]]
      [:tr
       [:td "Semantic Similarity"]
       [:td semantic-similarity]
       [:td (:semantic-similarity eval-averages-all)]
       [:td (make-indicator-symbol semantic-similarity
                                   (:semantic-similarity eval-averages-all))]]
      [:tr
       [:td "Recall"]
       [:td recall]
       [:td (:recall eval-averages-all)]
       [:td (make-indicator-symbol recall
                                   (:recall eval-averages-all))]]
      [:tr
       [:td "Precision"]
       [:td precision]
       [:td (:precision eval-averages-all)]
       [:td (make-indicator-symbol precision
                                   (:precision eval-averages-all))]]
      [:tr
       [:td "F1"]
       [:td f1]
       [:td (:f1 eval-averages-all)]
       [:td (make-indicator-symbol f1
                                   (:f1 eval-averages-all))]]]]))


(mapv #(kind/hiccup (model-performance-summary responses-eval-data %))
     (distinct (map :model-ref responses-eval-data)))


;; ### Evaluating the Evaluation Dataset
;;
;; As a last step, let's have a quick look to see if the metrics can tell us
;; anything about our evaluation dataset itself.
;;
;; For example, qhich question has the most wrong (non-faithfull) answers?

(-> responses-eval-data
    (tc/dataset)
    (tc/select-columns [:question :metric-llm-faithfulness-score])
    (tc/drop-missing :metric-llm-faithfulness-score) ;; There is actually one missing here...
    (tc/group-by [:question])
    (tc/aggregate {:total-correct #(apply + (% :metric-llm-faithfulness-score))})
    (tc/order-by :total-correct))

;; The question about healthcare assistants only had 11/20 correct answers. This
;; is unsurprising in retrospect, as even I had trouble understanding this
;; original question/answer.
;;
;; Let's look at a couple of examples/evaluation reasoning for the lowest-scoring question

(-> responses-eval-data
    (tc/dataset)
    (tc/select-columns [:model-ref :question :answer :metric-llm-faithfulness-score :metric-llm-faithfulness-explanation])
    (tc/drop-missing :metric-llm-faithfulness-score)
    (tc/select-rows #(and (= (:question %) "Will the government put in place Level 6 (QQI) courses for healthcare assistants?")
                          (= (:metric-llm-faithfulness-score %) 0)))
    (tc/select-columns [:model-ref :answer :metric-llm-faithfulness-explanation]))

;; We can see a major error here with my evaluation prompt. In some cases the
;; model answers that "It cannot provide information using the information
;; available" which should be an acceptable answer in this context (since the
;; prompt instructs is that it should provide this default if it can't answer)
;;
;; I went back and added an extra instruction in the prompt to try account for
;; these cases. But, it's an important lesson in trying to think logically about
;; the material in the prompts.

