(ns notebooks.make
  (:require [scicloj.clay.v2.api :as clay]))

(def book-spec
  {:format [:quarto :html]
   :book {:title "PQ Notebook"}
   :base-target-path "book"
   :base-source-path "src"
   :source-path ["index.clj"
                 "notebooks/preparation.clj"
                 "notebooks/question_vdb.clj"
                 "notebooks/vdb_evaluation.clj"
                 "notebooks/generation.clj"
                 "notebooks/rag_evaluation.clj"]
   :clean-up-target-dir true})

(defn make-book [_] (clay/make! book-spec))

(comment
  (make-book nil))
