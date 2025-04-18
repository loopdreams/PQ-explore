(ns notebooks.rag-evaluation
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py.] :as py]
            [notebooks.preparation :refer [ds]]
            [selmer.parser :as templates]
            [scicloj.kindly.v4.kind :as kind]
            [notebooks.generation :as gen]
            [scicloj.tableplot.v1.plotly :as plotly]
            [notebooks.llm-api :as llm]
            [clojure.edn :as edn]
            [clojure.string :as str]
            [tablecloth.api :as tc]
            [clojure.java.io :as io]))


;; Install:
;; - python3 -m pip install continuous-eval
;; - pip3 install torch torchvision torchaudio
;; - pip3 install pandas
;; (pytorch/pandas needed for semantic metrics)
;;
;; Available metrics:
;; - Deterministic
;;  - DeterministicAnswerCorrectness - requires ground truth answers
;;  - DeterministicFaithfulness
;;  - FleschKincaidReadability
;; - Llm-based
;;  - AnswerCorrectness
;;  - AnswerRelevance
;;  - Faithfulness
;;  - StyleConsistency
;; - Semantic
;;  - BertAnswerRelevance
;;  - BertAnswerSimilarity
;;  - DebertaAnswerScores


;; ## Generating/Saving LLM responses for evaluation


(defn ask-llm-save-responses! [model questions]
  (let [responses (reduce (fn [res question]
                            (conj res (gen/make-rag-data (assoc question :model-ref model))))
                          [] questions)
        f-name (str "data/responses/" model "_responses.edn")]
    (spit f-name responses)))

(def evaluation-dataset
  (let [data         (edn/read-string (slurp "data/evaluation_questions/questions.edn"))
        questions    (keys data)
        ground-truth (vals data)]
    (mapv (fn [question truth] (-> {}
                                   (assoc :question question)
                                   (assoc :ground-truth truth)))
          questions
          ground-truth)))

(comment
  (ask-llm-save-responses! "gemini-2.0-flash-lite" evaluation-dataset))


;; ## Continuous Eval Metrics
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
       "rouge_l_precision"           :rouge-1-precision
       "rouge_l_f1"                  :rouge-1-f1
       "rouge_p_by_sentence"         :rouge-p-by-sentence
       "bleu_score_by_sentence"      :bleu-score-by-sentence
       "bleu_faithfulness"           :bleu-faithfulness
       "bleu_score"                  :bleu-score
       "token_overlap_p_by_sentence" :token-overlap-p-by-sentence
       "token_overlap_f1"            :token-overlap-f1
       "token_overlap_precision"     :token-overlap-precision
       "token_overlap_recall"        :token-overlap-recall
       "token_overlap_faithfulness"  :token-overlap-faithfulness}))))

;; ### LLM Metrics

(defn add-llm-metric-correctness-score [{:keys [question answer ground-truth] :as rag-data} llm-model]
  (let [system-prompt (slurp "prompts/ans_correctness_sys.txt")
        user-prompt   (-> "prompts/ans_correctness_user.txt"
                          slurp
                          (templates/render {:question     question
                                             :answer       answer
                                             :ground-truth (if (seq ground-truth) ground-truth (:retrieved-context rag-data))}))
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


(defn add-all-evaluation-metrics [responses model]
  (mapv (fn [resp]
          (-> resp
              add-deterministic-metrics
              (add-llm-metrics model)))
        responses))

(defn run-and-save-evaluation-metrics! [responses model]
  (let [model-ref (:model-ref (first responses))
        f-name (str "data/responses_evaluation/" model-ref "_evaluation.edn")
        resp (add-all-evaluation-metrics responses model)]
    (spit f-name resp)))

(defn run-and-save-all-evals! [responses-dir model]
  (let [responses (->> (io/file responses-dir)
                       file-seq
                       rest
                       (mapv (comp edn/read-string slurp)))]
    (mapv #(run-and-save-evaluation-metrics! % model) responses)))

(comment
  (run-and-save-all-evals! "data/responses" "gpt-3.5-turbo"))

(defonce sample-metrics-2
  (add-all-evaluation-metrics (edn/read-string (slurp "data/responses/llama3.2_responses.edn"))
                              "llama3.2"))

(defonce sample-metrics-3
  (add-all-evaluation-metrics (edn/read-string (slurp "data/responses/gemini-2.0-flash-lite_responses.edn"))
                              "llama3.2"))


(defn average [coll]
  (float
   (/ (apply + (remove nil? coll))
      (count (remove nil? coll)))))

(defn average-all-cols [numerical-ds]
  (let [cols (tc/column-names numerical-ds)]
    (tc/dataset
     (reduce (fn [res col]
               (assoc res col (average (numerical-ds col))))
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

(kind/table
 (build-responses-eval-ds-avgs "data/responses_evaluation"))

(defn build-responses-eval-ds-all [responses-eval-dir]
  (let [responses (->> responses-eval-dir
                       io/file
                       file-seq
                       rest
                       (mapv (comp edn/read-string slurp)))
        ds (tc/dataset (reduce into responses))]
    (-> ds
        (tc/select-columns
         (concat
          (tc/column-names ds :type/numerical)
          [:model-ref :question])))))

;; ## Comparing the Metrics

(defn make-boxplot [metric]
  (->
   (build-responses-eval-ds-all "data/responses_evaluation")
   (tc/order-by metric)
   (plotly/layer-boxplot
    {:=x :model-ref
     :=y metric})))

;; ### Deterministic Metrics (non-llm)
;; #### Reading Ease
;;
;; The `flesch-kincaid-grade-level` and `flesch-reading-ease` metrics help show
;; how readible the response is. A lower grade level and higher reading ease
;; level makes the text more readible.
;;
;; As a reference, let's calculate the avereage grade-level and reading-ease
;; for the actual responses provided by departments.

(def reading-scores
  (->> (:answer (tc/drop-missing ds :answer))
       (mapv #(py.. (det/FleschKincaidReadability) (compute %)))
       (mapv vals)))

;; Reading Ease Average
(def baseline-reading-ease
  (average (map first reading-scores)))

baseline-reading-ease

;; Grade Level Average
(def baseline-grade-level
  (average (map second reading-scores)))

baseline-grade-level

;; These scores indicate that the pre-exiting answers are quite difficult to read.

(make-boxplot :flesch-reading-ease)

(make-boxplot :flesch-kincaid-grade-level)


;; ### LLM Generated Metrics
;; #### Faithfulness

(make-boxplot :metric-llm-faithfulness-score)


;; #### Correctness

(make-boxplot :metric-llm-correctness-score)

;; #### Relevance

(make-boxplot :metric-llm-relevance-score)



;; ### 'Overall' Rating
;; TODO: some kind of weighing scheme to highlight which metrics impact outcome most
