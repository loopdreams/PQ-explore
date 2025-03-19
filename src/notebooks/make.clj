(ns notebooks.make
  (:require [scicloj.clay.v2.api :as clay]))

(def book-spec
  {:format [:quarto :html]
   :book {:title "PQ Notebook"}
   :base-target-path "book"
   :base-source-path "src/notebooks"
   :source-path ["preparation.clj"
                 "question_vdb.clj"
                 "rag.clj"]
   :clean-up-target-dir true})

(defn make-book [_] (clay/make! book-spec))

(comment
  (make-book nil))
