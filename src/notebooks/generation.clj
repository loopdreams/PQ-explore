(ns notebooks.generation
  (:require
   [clojure.string :as str]
   [notebooks.llm-api :as llm]
   [notebooks.preparation :refer [ds]]
   [scicloj.kindly.v4.kind :as kind]
   [notebooks.vdb-evaluation :as vdb]
   [tablecloth.api :as tc]
   [java-time.api :as jt]))

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

(defn retrieve-context [question]
  (let [related-docs (mapv :text (vdb/generate-context question))]
    (str/join "\n\n" related-docs)))

(defn generate-prompt [question]
  (let [retrieved-context (retrieve-context question)]
    (str "I want you to act as a responsible and trustworthy senior government official.
Please provide an answer to a citizen's question, using only the context provided.
Answer as if you are talking directly to the citizen and be neutral and formal as possible.
If you can't find a specific detail from the question, please acknowledge this and provide any
other helpful information that may be related to the question.
If you can't find sufficient information in the context to answer the question at all,
then reply with \"I am unable to answer this question with the information I have available.\""
         "\n\n CONTEXT: " retrieved-context)))


(kind/md
 (generate-prompt sample-question))


;; Now that we have a way to generate a prompt, let's pass it to an LLM.
;;
;; [text here referencing the llm namespace]

(kind/table llm/llm-models)


(comment
  (kind/md
   (llm/ask-llm
    {:question sample-question
     :system-prompt (generate-prompt sample-question)
     :model-ref "llama3.2"})))


;; Let's try a few additional initial tests:

(def sample-question-broad "What is the government doing to support GPs?")
(def sample-question-detail "What is the government doing to support GPs in Limerick city?")

(comment
  (kind/md
   (llm/ask-llm
    {:question sample-question-broad
     :system-prompt (generate-prompt sample-question-broad)
     :model-ref "llama3.2"}))

  (kind/md
   (llm/ask-llm
    {:question sample-question-detail
     :system-prompt (generate-prompt sample-question-detail)
     :model-ref "llama3.2"})))


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

(defn generate-answer-with-references [llm-config]
  (let [context-docs (mapv :text (vdb/generate-context (:question llm-config)))
        docs-links (format-links-md (mapv get-reference-link-for-doc context-docs))
        llm-response (llm/ask-llm llm-config)]
    (str llm-response "\n\n" docs-links)))

(comment
  (kind/md
   (generate-answer-with-references
    {:question sample-question-broad
     :system-prompt (generate-prompt sample-question-broad)
     :model-ref "llama3.2"})))
