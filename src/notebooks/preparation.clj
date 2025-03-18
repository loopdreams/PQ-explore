;; # Dataset Preparation
(ns notebooks.preparation
  (:require [tablecloth.api :as tc]
            [clojure.string :as str]))



;; ## Some cleaning/preparation steps on the dataset.

(def datasource "data/20250302_PQs_10K_2024_answers.csv")


;; ### Text Cleaning
;;
;; #### Question Formatting
;; The questions are prefixed with a question number, and end
;; with an id tag. The functions below aim to remove these.
;;
;; "1. An example question? [1234/25]" ->> "An example question?"

(def re-question-number  "^\\d+. ")
(def re-question-id   "\\[\\d+/\\d+\\]")
(def re-question-num-or-id (re-pattern (str re-question-id "|" re-question-number)))

(defn clean-question [q] (str/replace q re-question-num-or-id ""))

;; #### Topic Labels
;; Some topic labels contain a trailing period at the end. We
;; will also remove these.

(defn clean-topic-label [label]
  (when label
    (if (re-find #"\.$" label)
      (subs label 0 (dec (count label)))
      label)))

;; #### Department Names
;; At various times, especially following general elections, department functions
;; can change. This typically also involves a change in the deparment's name.
;;
;; Because of this, it is hard to track most deparments consistently beyond the last five years
;; or so. Some departments, such as 'Health' or 'Justice' remain largely the same.
;;
;; In addition, older questions give the full department title, while more recent
;; questions only give the first part of the title. For example, "Department for the Environment, Climate and Communications"
;; becomes "Environment".
;;
;; In order to try consolidate some of the department names, we will also transform
;; the older labels into single-word department names.
;;
;; TODO: Also, map an additional column with the 'full name' of the department. At the
;; moment (following a recent election), this full names are still being processed/decided,
;; so come back to this later.

(defn normalise-department-name [label]
  (cond
    (re-find #"^Minister for Expenditure" label) "Public Expenditure"
    (re-find #"^Public$" label) "Public Expenditure"
    (re-find #"^Minister for the" label) (first (re-find #"(?<=^Minister for the )(\w+)(?=,| |$)" label)) ;; To match "Minister for the Environment..."
    (re-find #"^Minister for" label) (first (re-find #"(?<=^Minister for )(\w+)(?=,| |$)" label))
    :else label))


;; ## Build Prepared Dataset

(defonce ds
  (-> datasource
      (tc/dataset {:key-fn keyword})
      (tc/map-columns :question [:question] clean-question)
      (tc/map-columns :topic [:topic] clean-topic-label)
      (tc/map-columns :department [:department] normalise-department-name)
      (tc/select-columns [:date :question :answer :member :department :topic :house])))

(tc/head ds)

(tc/tail ds)
