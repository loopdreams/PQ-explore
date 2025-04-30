;; # Dataset Preparation
(ns notebooks.preparation
  (:require [tablecloth.api :as tc]
            [java-time.api :as jt]
            [clojure.string :as str]
            [scicloj.kindly.v4.kind :as kind]))

;; ## Cleaning/preparation steps

;; The data is taken from the [Oireachtas website](https://www.oireachtas.ie/).
;; It contains key fields such as 'question', 'answer', 'date' and 'topic'.
;; There are around 10K quesitons/answers in the initial dataset, but many of
;; these will be removed through some cleaning steps (below).
;;
;; The questions/answers are written submissions by members of parliament on a
;; wide variety of topics. The written answers are provided by Ministers, who
;; are heads of various departments.

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

(defn normalise-department-name [label]
  (cond
    (re-find #"^Minister for Expenditure" label) "Public Expenditure"
    (re-find #"^Public$" label) "Public Expenditure"
    (re-find #"^Minister for the" label) (first (re-find #"(?<=^Minister for the )(\w+)(?=,| |$)" label)) ;; To match "Minister for the Environment..."
    (re-find #"^Minister for" label) (first (re-find #"(?<=^Minister for )(\w+)(?=,| |$)" label))
    :else label))

;; #### Answer Cleaning
;; The data for the question 'answers' was in xml format, and occasionally
;; included things like table elements. While parsing these I ommotted them and
;; left the string '{{OMMITTED ...}}' in their place. So, I will also add a step
;; here to remove those parts of the string.

(defn clean-incomplete-answers [answer]
  (str/replace answer #"\{\{OMITTED.*element\}\}" ""))

;; Some answers also contain the 'non-breaking space' character (ascii code
;; 160), so we will try to replace these with spaces.

(defn clean-nbs-answers [answer]
  (str/replace answer #" " " "))

;; ### Duplicate questions

;; There are some questions that are duplicates. For example:

(kind/table
 (->> (tc/map-columns (tc/dataset datasource {:key-fn keyword}) :question [:question] clean-question)
      :question
      (frequencies)
      (sort-by second)
      reverse
      (take 2)))

;; You can see from these that the issue is because there are separate details supplied that are not available here.
;;
;; For the purposes of this exercise, it is better to remove these duplicates entirely, and we will do so below
;; using tablecloth's unique-by function.


;; ### Adding Question URLs
;;
;; In case we want to reference the original source, we'll also add the question
;; urls to the dataset.

(defn extract-question-num [q] (re-find #"^\d+" q))
(defn extract-question-id [q] (re-find #"(?<=\[).*(?=\])" q))

(defn make-url [date q-num]
  (let [url-base "https://www.oireachtas.ie/en/debates/question/"
        url-default "https://www.oireachtas.ie/"]
    (if (jt/< (jt/local-date date) (jt/local-date "2012-07-01"))
      (str url-default)
      (str url-base (str date) "/" q-num "/"))))

;; ## Build Prepared Dataset

(def ds
  (-> datasource
      (tc/dataset {:key-fn keyword})
      (tc/drop-missing :answer)
      (tc/map-columns :q-num [:question] extract-question-num)
      (tc/map-columns :q-id [:question] extract-question-id)
      (tc/map-columns :url [:date :q-num] #(make-url %1 %2))
      (tc/map-columns :question [:question] clean-question)
      (tc/map-columns :topic [:topic] clean-topic-label)
      (tc/map-columns :department [:department] normalise-department-name)
      (tc/drop-missing :answer)
      (tc/map-columns :answer [:answer] (comp clean-incomplete-answers clean-nbs-answers))
      (tc/unique-by :question)
      (tc/select-columns [:date :question :answer :department :topic :url])))

;; ## General Stats

^:kindly/hide-code
(def total-questions (tc/row-count ds))

^:kindly/hide-code
(def ds-date-start (-> ds (tc/order-by :date) :date first))

^:kindly/hide-code
(def ds-date-end (-> ds (tc/order-by :date :desc) :date first))

^:kindly/hide-code
(def number-topics (-> ds :topic distinct count))

^:kindly/hide-code
(def top-5-topics (take 5 (-> ds (tc/group-by [:topic])
                              (tc/aggregate {:count tc/row-count})
                              (tc/order-by :count :desc)
                              :topic)))

^:kindly/hide-code
(def top-5-most-asked-departments (take 5 (-> ds (tc/group-by [:department])
                                              (tc/aggregate {:count tc/row-count})
                                              (tc/order-by :count :desc)
                                              :department)))


^:kindly/hide-code
(kind/hiccup
 [:div
  [:ul
   [:li "Dates range from "
    [:strong (jt/format "MMMM dd yyyy" ds-date-start)] " to "
    [:strong (jt/format "MMMM dd yyyy" ds-date-end)]]
   [:li [:strong (format "%,d" total-questions)] " total questions asked."]
   [:li "The five most common question topics are: " (str/join ", " top-5-topics)]
   [:li "The five most commonly asked departments are: " (str/join ", " top-5-most-asked-departments)]]])


;; ## A quick look at the dataset

(tc/column-names ds)

(tc/row-count ds)

(tc/head ds)
