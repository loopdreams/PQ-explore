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
            [clojure.java.io :as io]))


;; **NOTE!**
;; This section uses clojure-python interop. In order for it to load properly,
;; you'll have to set up libpython-clj2 with your editor and install the
;; relevant python dependencies seperately.
;;
;; TODO: test this with a fresh clone of repo
;; In terms of the python dependencies, the follow should be all that is needed:
;;
;; `python3 -m pip install continuous-eval`
;;
;; In my case, I set up a python virtual environment to install this, and then created
;; a file under 'dev/user.clj' file that is automatically run by emacs/cider when starting a REPL.
;; Inside this file is just the 'initialize!' function provided by libpython-clj2.python namespace,
;; which takes a path to a :python-executable and a :library-path

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

;; # Rag Evaluation
;;
;; ## Overview
;; For this section, I will be relying heavily on the [continuous-eval (python)](https://github.com/relari-ai/continuous-eval)
;; metrics and approach for starting to think about how to evaluate the RAG.
;;
;; That repo also has some great links to articles explaining some of the
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
;; In this exercises, I will only really look at the question of what llm
;; **model** might work best with the data that I have and the prompt/retrieval
;; framework we have already set up.
;;
;; We will focus on two categories of metric:
;;
;; - Deterministic
;;
;; - LLM-based
;;
;; Deterministic metrics are similar to how we measured the retrieval
;; performace; they simple measure the *token overlap* between answers generated
;; by the LLM and some kind of reference/ground-truth answers.
;;
;; LLM-based metrics utilise another LLM to assign a score to the output. For
;; example, to determine 'answer-correctness', we will ask an LLM to assign a
;; score between 1-5 to a generated answer, based on reference answers that we
;; provide ourselves.
;;
;; Before going into the metrics further, we will first create:
;;
;; - A testing dataset, contining some questions and ground truth answers
;;
;; - Some generated LLM responses by different models, using the questions from
;;   the testing dataset
;;
;; For the testing dataset, I've used 10 fairly random questions based on some
;; of the material in the starting dataset of questions and answers. It is saved
;; in a 'questions.edn' file in this project.
;;
;; Ideally, we would use a much larger and more thoughfully curated evaluation
;; dataset, perhaps with input from domain experts across different question areas.
;; The goal here, however, is simply to test out some evaluation workflows in
;; clojure, so a basic evaluation dataset will have to do for now. Below, we
;; just load that dataset. The 'questions.edn' file is set up as a clojure map,
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

;; Next, we will write a helper function to save llm responses and generate some
;; responses by different llm models.

(defn ask-llm-save-responses! [model questions]
  (let [responses (reduce (fn [res question]
                            (conj res (gen/get-rag-answer (assoc question :model-ref model))))
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
  (ask-llm-save-responses! "gemma3:4b" evaluation-dataset)
  (ask-llm-save-responses! "granite3.2" evaluation-dataset)
  (ask-llm-save-responses! "gemini-2.5-pro-preview-03-25" evaluation-dataset)
  (ask-llm-save-responses! "gemini-2.5-flash-preview-04-17" evaluation-dataset))

(def responses-ds
  (let [responses-dir "data/responses"
        responses (->> responses-dir
                       (io/file)
                       file-seq
                       rest
                       (map (comp edn/read-string slurp))
                       (reduce into))]
    (tc/dataset responses)))

(tc/row-count responses-ds)

;; ## Continuous Eval Metrics Functions
;;
;; Below, I am just creating a wrapper for the Continuous-eval deterministic
;; metrics, and re-writing the LLM metrics in clojure, using the
;; [prompt templates that are provided in the continuous-eval repo](https://github.com/relari-ai/continuous-eval/tree/main/continuous_eval/metrics/generation/text/prompts)
;;
;; For demonstrating how the metrics work, we will use a couple of the generated responses as samples.
;;
;; For the question "How many households were in reciept of HAP payments in
;; 2023?", the data available states that 57,617 households were in receipt of
;; payments at the end of **Q3 2023**. In other words, the full data for 2023
;; was not available at that time. Most of the models seemed to be able to pick
;; up that detail, but one of the lower-powered ones, gemma3(1 billion param
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


;
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

;; Example score for the sample responses:

(-> (mapv add-deterministic-metrics (tc/rows sample-gen-responses :as-maps))
    (tc/dataset)
    (tc/select-columns [:model-ref :question :answer :rouge-1-f1 :token-overlap-f1 :bleu-score])
    (kind/table))

;; The 'F1' scores are the combination of 'precision' and 'recall' metrics. As
;; we saw in previous sections, precision is how much of the generated asnwer is
;; reflected in the ground truth (i.e., what % of the generated answer is
;; 'superfluous'), and recall is how much of the ground truth is reflected in
;; the generated answer. The F1 score is the harmonic mean of both these scores,
;; with a score closer to 1 being better. The 'BLEU' score is also better when
;; it is closer to 1
;;
;; In this case, even though these metrics don't check for semantic meaning or
;; logic, the metrics do indicate that the llama3.1 responses were slightly
;; better than the gemma3 responses.


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
        resp (add-all-evaluation-metrics responses model)]
    (spit f-name resp)))

(defn run-and-save-all-evals! [responses-dir model]
  (let [responses (->> (io/file responses-dir)
                       file-seq
                       rest
                       (mapv (comp edn/read-string slurp)))]
    (mapv #(run-and-save-evaluation-metrics! % model) responses)))

(comment
  ;; 43:55 (very roughly) to run around 15
  ;; cost - around 1.44 USD for 18 models * 10 questions each - 180 evaluations
  (run-and-save-all-evals! "data/responses" "o4-mini-2025-04-16"))


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

(defn concat-responses-eval-data [responses-eval-dir]
  (let [responses (->> responses-eval-dir
                       io/file
                       file-seq
                       rest
                       (mapv (comp edn/read-string slurp)))]
    (reduce into responses)))

(defn concat-responses-eval-ds-narrowed [responses-eval-dir]
  (let [ds (tc/dataset (concat-responses-eval-data responses-eval-dir))]
    (-> ds
        (tc/select-columns
         (concat
          (tc/column-names ds :type/numerical)
          [:model-ref :question])))))

;; ## Comparing the Metrics

(def responses-eval-data (concat-responses-eval-data "data/responses_evaluation"))
(def responses-eval-ds-narrowed (concat-responses-eval-ds-narrowed "data/responses_evaluation"))

(defn make-boxplot [metric]
  (->
   responses-eval-ds-narrowed
   (tc/order-by :model-ref)
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

(def precision-grouped-by-question
  (-> responses-eval-ds-narrowed
      (tc/select-columns [:question :model-ref :rouge-1-precision :token-overlap-precision])
      (tc/group-by :question)
      :data))



(defn graph-group [ds key]
  (-> (tc/order-by ds key)
      (plotly/base
       {:=x :model-ref
        :=title (first (:question ds))})
      (plotly/layer-line
       {:=y key})
      (plotly/layer-line
       {:=y :token-overlap-precision})))

(map #(graph-group % :rouge-1-precision) precision-grouped-by-question)


;; Recall

(def recall-grouped
  (-> responses-eval-ds-narrowed
      (tc/select-columns [:question :model-ref :rouge-l-recall :token-overlap-recall])
      (tc/group-by :question)
      :data))

(map (fn [ds]
       (->
        (tc/order-by ds :rouge-l-recall)
        (plotly/base
         {:=x :model-ref
          :=title (first (:question ds))
          :=width 800})
        (plotly/layer-line
         {:=y :rouge-l-recall})
        (plotly/layer-line
         {:=y :token-overlap-recall})))
     recall-grouped)

;; Precision/Recall

(def multiple-grouped
  (-> responses-eval-ds-narrowed
      (tc/order-by :model-ref)
      (tc/group-by :question)
      :data))

(map (fn [ds]
       (-> ds
           (plotly/base
            {:=x :model-ref
             :=title (first (:question ds))
             :=width 800})
           (plotly/layer-bar
            {:=y :bleu-score})
           (plotly/layer-bar
            {:=y :rouge-1-f1})
           (plotly/layer-bar
            {:=y :token-overlap-f1})))
     multiple-grouped)

(defn add-model-detail [ds detail]
  (-> ds
      (tc/map-columns detail [:model-ref]
                      (fn [m]
                        (->
                         (filter (fn [m1] (= m (:model-ref m1))) llm/llm-models)
                         first
                         detail)))))

(->
 (build-responses-eval-ds-avgs "data/responses_evaluation")
 (add-model-detail :platform)
 (tc/order-by :rouge-1-f1)
 (plotly/base
  {:=width 800})
 (plotly/layer-bar
  {:=x :model-ref
   :=y :rouge-1-f1
   :=color :platform}))


;; ### LLM Generated Metrics
;; #### Faithfulness

(defn make-bar-avgs [metric]
  (->
   (build-responses-eval-ds-avgs "data/responses_evaluation")
   (tc/order-by metric)
   (plotly/layer-bar
    {:=x :model-ref
     :=y metric})))

(make-bar-avgs :metric-llm-faithfulness-score)


;; #### Correctness

(make-bar-avgs :metric-llm-correctness-score)

;; #### Relevance

(make-bar-avgs :metric-llm-relevance-score)


;; #### Single Model Scores

;; Matrix:
;;              Score | Average (all models) |
;; Recall     |
;; Precision
;; Faithfulness
;; Correctness
;; Relevance
;;
;; TODO: calculate baseline averages
;; - across score
;; - per question


;; ### 'Overall' Rating
;; TODO: some kind of weighing scheme to highlight which metrics impact outcome most
