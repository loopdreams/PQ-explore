(ns notebooks.generation
  (:require
   [clojure.string :as str]
   [notebooks.llm-api :as llm]
   [notebooks.preparation :refer [ds]]
   [scicloj.kindly.v4.kind :as kind]
   [notebooks.vdb-evaluation :as vdb]
   [tablecloth.api :as tc]))

;; ## Retrieval Augmented Generation
;; [Intro text]

;; To generate an answering using a RAG approach, all we have to do is add our
;; retrieved context to a prompt.
;;
;; We will re-use the same question as we used in the previous
;; section.

(def sample-question "How much annual investment was provided under the 2019 GP agreement?")

;; Next, we will create some prompt functions to help build the context.
;;
;; Before passing, the question to the prompt generator, we will re-use the function in the last section to generate the context.


(defn add-context [{:keys [question] :as rag-data} db-store]
  (let [ctx (mapv :text (vdb/generate-context question db-store))]
    (assoc rag-data :retrieved-context ctx)))

(defn add-pq-prompt [{:keys [retrieved-context] :as rag-data}]
  (assoc rag-data :system-prompt
         (str "I want you to act as a responsible and trustworthy senior government official.
Please provide an answer to a citizen's question, using only the context provided.
Answer as if you are talking directly to the citizen and be neutral and formal as possible.
If you can't find a specific detail from the question, please acknowledge this and provide any
other helpful information that may be related to the question.
If you can't find sufficient information in the context to answer the question at all,
then reply with \"I am unable to answer this question with the information I have available.\""
              "\n\n CONTEXT: " (str/join "\n\n" retrieved-context))))


(kind/md
 (-> {:question sample-question}
     (add-context vdb/db-store-chunked-answers)
     add-pq-prompt
     :system-prompt))

;; Now that we have a way to generate a prompt, let's pass it to an LLM.
;;
;; [text here referencing the llm namespace]

(kind/table llm/llm-models)

(defn add-llm-response [{:keys [model-ref question system-prompt] :as rag-data}]
  (let [answer (llm/ask-llm
                {:model-ref model-ref
                 :question question
                 :system-prompt system-prompt})]
    (assoc rag-data :answer answer)))

(comment
  (-> {:question sample-question
       :model-ref "llama3.2"}
      add-context
      add-pq-prompt
      add-llm-response
      :answer
      kind/md))



;; Let's try a few additional initial tests:

(def sample-question-broad "What is the government doing to support GPs?")
(def sample-question-detail "What is the government doing to support GPs in Limerick city?")

(comment
  (-> {:question sample-question-broad
       :model-ref "llama3.2"}
      add-context
      add-pq-prompt
      add-llm-response
      :answer
      kind/md)


  (-> {:question sample-question-detail
       :model-ref "llama3.2"}
      add-context
      add-pq-prompt
      add-llm-response
      :answer
      kind/md))



;; Putting these together into single function:

(defn get-rag-answer [rag-data db-store]
  (-> rag-data
      (add-context db-store)
      add-pq-prompt
      add-llm-response))



;; As an added feature, we could use the original dataset to try provide a reference for the question.


(defn get-reference-link-for-doc [doc]
  (-> ds
      (tc/drop-missing :answer)
      (tc/select-rows #(re-find (re-pattern doc) (:answer %)))
      :url
      first))

(defn format-links-md [links]
  (str "*References* \n\n- "
       (str/join "\n\n- "
                 links)))

(defn generate-answer-with-references [rag-data db-store]
  (let [rag-data (get-rag-answer rag-data db-store)
        ctx-docs (:retrieved-context rag-data)
        docs-ref-links (format-links-md (mapv get-reference-link-for-doc ctx-docs))]
    (str (:answer rag-data) "\n\n" docs-ref-links)))

(comment
  (-> {:question sample-question
       :model-ref "gemini-2.0-flash"}
      (generate-answer-with-references vdb/db-store-chunked-answers)
      kind/md))
