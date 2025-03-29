;; # Questions Vector Database
(ns notebooks.question-vdb
  (:require [tablecloth.api :as tc]
            [notebooks.preparation :refer [ds]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [clojure.string :as str]
            [jsonista.core :as json]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.core :as mm]
            [libpython-clj2.python :refer [py.]]
            [libpython-clj2.require :refer [require-python]])
  (:import (dev.langchain4j.data.embedding Embedding)
           (dev.langchain4j.data.segment TextSegment)
           (dev.langchain4j.model.embedding EmbeddingModel)
           (dev.langchain4j.store.embedding EmbeddingMatch)
           (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
           (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)
           (smile.manifold TSNE)
           (smile.feature.extraction PCA)))

;; ## Introduction [Intro text]


;; ## Building a Vector Database from the Questions

;; We will first use langchain4j to transform question text into vector embeddings. The model
;; we will use is provided by langchain4j.

(def model (AllMiniLmL6V2EmbeddingModel/new))

(def questions-list (:question ds))

;; Let's look at a single embedding for reference.

(->> (TextSegment/from (first questions-list))
     (. model embed)
     (.content))

;; As you can see, it is a very long vector of floating points representing the text.
;;
;; In order to store the embeddings in a database, we will use an in-memory database provided
;; by langchain4j. There are also options for using a more robust solution like postgres, but
;; for testing/exploration purposes, an in-memory database will do.


;; TODO: These are stored as file to avoid always loading them. For final version uncomment these.
(comment
  (def db-store (InMemoryEmbeddingStore/new)))

;; A short function for adding a question to the store.

(defn add-question-to-store! [question store]
  (let [segment (TextSegment/from question)
        embedding (->> segment (. model embed) (.content))]
    (. store add embedding segment)))

;; Finally, adding all the questions to the store. The mapping, which has side effects, is wrapped in a 'count' function
;; just so we can see at the end how many questions have been added.

(comment
  (count (map #(add-question-to-store! % db-store) questions-list)))

(comment
  (spit "data/db-store-questions.json" (.serializeToJson db-store)))

(def db-store (InMemoryEmbeddingStore/fromFile "data/db-store-questions.json"))

;; ## Testing Lookup

;; Next, let's try to use the database to return similar questions from the database.

(def test-question (first questions-list))

test-question

;; It's a very generic question...

(defn query-db-store [text n]
  (let [query (.content (. model embed text))
        result (. db-store findRelevant query n)]
    (map (fn [entry]
           {:text (.text (.embedded entry))
            :score (.score entry)})
         result)))

(kind/table
 (query-db-store test-question 5))

;; Let's try with a made-up question

(kind/table
 (query-db-store "The Deputy would like to ask the Minister about schemes relating to climate change." 5))

;; Finally, let's also try with a more focused question

(-> ds
    (tc/select-rows #(= (% :topic) "Dublin Bus"))
    :question
    first
    kind/md)

(def test-question-2 "The Deupty asked the Minister about Dublin Bus Route 74 and what their plans for this route are.")

(kind/table
 (query-db-store test-question-2 5))

;; Two of the five questions relate to the national bus service (Bus Eireann) as opposed to Dublin Bus, but overall
;; it is quite close.

;; ## Answering a Question
;;
;; Let's finally use this method to return the best answers based on the question given. This
;; approach could be later used to provide context for a RAG model that would generate answers.

(defn get-answers-for-question [question]
  (let [matching-questions (map :text (query-db-store question 5))]
    (-> ds
        (tc/select-rows #(some #{(% :question)} matching-questions))
        (tc/select-columns [:answer :date]))))

(kind/table
 (get-answers-for-question test-question-2))


;; Using just chatgpt, here is how a sample answer might possibly be generated:
;;
;; > The Minister stated that while he is responsible for public transport policy and funding, the day-to-day operations, including Dublin Bus Route 74, fall under the National Transport Authority (NTA). He has therefore referred the Deputy’s question to the NTA for a direct reply.

;; ## Question Similarity Across the Database
;;
;; For this exploration, we will take the first 100 questions, and count how
;; many questions are in the database based on similarity

(def test-questions (take 100 questions-list))


;; ### Similarity by Topic

;; We'll look at the top 7 main topics

(def top-7-topics
  (-> (tc/group-by ds [:topic])
      (tc/aggregate {:count tc/row-count})
      (tc/order-by :count [:desc])
      (tc/select-rows (range 7))))

top-7-topics

;; Next, let's try to write a function to count the level of similarity among a topic group.
;;
;; The idea here is to:
;;
;; 1. Create a mini-store of all the questions related to a topic
;;
;; 2. Check each question against this store, and count it if it has at least 1 (excluding itself) question that is similar to it, based on a threshold
;;
;; 3. Calculate the percentage of questions that have at least one other similar question

(defn percentage-similar-questions [store questions threshold]
  (let [qs-above-threshold
        (reduce (fn [res question]
                  (let [query (.content (. model embed question))
                        matches (. store findRelevant query 100)]
                    (conj res
                          (-> (filter (fn [match] (> (.score match) threshold)) matches)
                              count
                              (- 1)))))
                [] questions)
        count-qs-above-threshold (count (remove zero? qs-above-threshold))]
    (float (/ count-qs-above-threshold (count questions)))))


(defn category-similarity [name type threshold]
  (let [target-questions (-> ds (tc/select-rows #(= (type %) name)) :question)
        topic-store (InMemoryEmbeddingStore/new)
        add-question (fn [q]
                       (let [seg (TextSegment/from q)
                             emb (->> seg (. model embed) (.content))]
                         (. topic-store add emb seg)))]
    (do
     (mapv add-question target-questions)
     (percentage-similar-questions topic-store target-questions threshold))))


(let [topics (:topic top-7-topics)
      data (mapv (fn [topic] (let [score (category-similarity topic :topic 0.9)]
                               {:topic topic
                                :score score}))
                 topics)]
  (-> (tc/dataset data)
      (plotly/layer-bar
       {:=x :topic
        :=y :score})))

;; At a lower threshold the topics are highly similar:

(let [topics (:topic top-7-topics)
      data (mapv (fn [topic] (let [score (category-similarity topic :topic 0.75)]
                               {:topic topic
                                :score score}))
                 topics)]
  (-> (tc/dataset data)
      (plotly/layer-bar
       {:=x :topic
        :=y :score})))


;; Based on the above, of the top 7 topics, the 'Health Services' questions seem the most diverse, which makes sense.

;; We can do a similar kind of comparison with the top seven departments

(def top-7-departments
  (-> ds
      (tc/group-by [:department])
      (tc/aggregate {:count tc/row-count})
      (tc/order-by :count :desc)
      (tc/select-rows (range 7))))

top-7-departments

;; (let [departments (:department top-7-departments)
;;       data (mapv (fn [department] (let [score (category-similarity department :department 0.9)]
;;                                     {:department department
;;                                      :score score}))
;;                  departments)]
;;   (-> (tc/dataset data)
;;       (plotly/layer-bar
;;        {:=x :department
;;         :=y :score})))

;; How about some example questions that aren't similar to other questions?

(defn outlier-questions
  "Looks for the most similar questions (excluding self) and checks if score is below threshold."
  [topic threshold]
  (let [target-questions (-> ds (tc/select-rows #(= (:topic %) topic)) :question)]
    (filter (fn [question]
              (let [query (.content (. model embed question))
                    match (. db-store findRelevant query 2)]
                (< (.score (second match)) threshold)))
            target-questions)))


(->> 0.80
     (outlier-questions "Disability Services")
     (str/join "\n\n")
     kind/md)

(->> 0.77
     (outlier-questions "Health Services")
     (str/join "\n\n")
     kind/md)

(->> 0.82
     (outlier-questions "Schools Building Projects")
     (str/join "\n\n")
     kind/md)


;; Question Similarity Curves


;; (defn category-similarity [name type threshold]
;;   (let [target-questions (-> ds (tc/select-rows #(= (type %) name)) :question)
;;         topic-store (InMemoryEmbeddingStore/new)
;;         add-question (fn [q]
;;                        (let [seg (TextSegment/from q)
;;                              emb (->> seg (. model embed) (.content))]
;;                          (. topic-store add emb seg)))]
;;     (do
;;      (mapv add-question target-questions)
;;      (percentage-similar-questions topic-store target-questions threshold))))

(def db-store-ip (InMemoryEmbeddingStore/new))

(let [ip-questions (:question (tc/select-rows ds #(= (:topic %) "International Protection")))]
  (count
   (mapv (fn [q]
           (let [seg (TextSegment/from q)
                 emb (->> seg (. model embed) (.content))]
             (. db-store-ip add emb seg)))
         ip-questions)))

(def test-question-3 (first (:question (tc/select-rows ds #(= (:topic %) "International Protection")))))

(defn all-similarity-scores [question store & ignore-self?]
  (let [emb (.content (. model embed question))
        result (. store findRelevant emb 200)]
    (map-indexed (fn [idx e] {:idx idx :score (.score e)})
                 (if ignore-self? (rest result) result))))

(-> (all-similarity-scores test-question-3 db-store-ip true)
    (tc/dataset)
    (plotly/layer-line
     {:=x :idx
      :=y :score}))

(defn plot-n-random-scores [topic store n]
  (let [candidates (:question (tc/select-rows ds #(= (:topic %) topic)))
        selected (take n (repeatedly #(rand-nth candidates)))
        data (loop [n 0 result []]
               (if (= n (count selected)) result
                   (let [q (nth selected n)
                         scores (all-similarity-scores q store true)
                         scores (map #(assoc % :question n) scores)]
                     (recur (inc n) (into result scores)))))]
    (tc/dataset data)))

(-> (plot-n-random-scores "International Protection" db-store-ip 1)
    (plotly/base
     {:=x :idx
      :=y :score})
    plotly/layer-point
    plotly/layer-smooth)

(-> (plot-n-random-scores "International Protection" db-store-ip 20)
    (plotly/base
     {:=x :idx
      :=y :score})
    plotly/layer-point
    plotly/layer-smooth)

(-> (plot-n-random-scores "International Protection" db-store-ip 50)
    (plotly/base
     {:=x :idx
      :=y :score})
    plotly/layer-point
    plotly/layer-smooth)

(-> (plot-n-random-scores "International Protection" db-store-ip 50)
    (plotly/layer-histogram
     {:=x :score}))

(-> (plot-n-random-scores "International Protection" db-store-ip 1)
    (plotly/layer-histogram
     {:=x :score}))

;; For comparison, let's try an "International Protection" question
;; within a different topic area


(def db-store-schools (InMemoryEmbeddingStore/new))

(let [schools-questions (:question (tc/select-rows ds #(= (:topic %) "Schools Building Projects")))]
  (count
   (mapv (fn [q]
           (let [seg (TextSegment/from q)
                 emb (->> seg (. model embed) (.content))]
             (. db-store-schools add emb seg)))
         schools-questions)))

(def random-ip-question (rand-nth (:question (tc/select-rows ds #(= (:topic %) "International Protection")))))

(def scores-self-group-other-group
  (let [scores-ip (all-similarity-scores random-ip-question db-store-ip true)
        scores-ip (map (fn [s] (assoc s :batch "same-group")) scores-ip)
        scores-sch (all-similarity-scores random-ip-question db-store-schools)
        scores-sch (map (fn [s] (assoc s :batch "other-group")) scores-sch)]
    (-> (concat scores-ip scores-sch)
        (tc/dataset))))

(-> scores-self-group-other-group
    (plotly/base
     {:=x :idx
      :=y :score
      :=color :batch})
    (plotly/layer-point)
    (plotly/layer-smooth))

(-> scores-self-group-other-group
    (plotly/layer-histogram
     {:=x :score
      :=color :batch
      :=mark-opacity 0.7}))

;; ## Visualising with tSNE

(def test-data {:a [0.1 0.4 0.2] :b [0.3 0.9 0.8]})

(def test-data-doubles
  (-> test-data
      (tc/dataset)
      (tc/rows :as-double-arrays)))

(def tsne (TSNE. test-data-doubles 2 20 200 100))

(->
 (. tsne coordinates)
 (tc/dataset))


(plotly/layer-point (-> (. tsne coordinates)
                        (tc/dataset))
  {:=x 0
   :=y 1})

(defn smile-minst->arr [str]
  (let [lines (str/split-lines str)]
    (reduce (fn [res line]
              (conj res
                    (let [str-nums (str/split line #"\ ")]
                      (mapv parse-double str-nums))))
            []
            lines)))

(defonce smile-mnist (smile-minst->arr (slurp "https://raw.githubusercontent.com/haifengl/smile/refs/heads/master/shell/src/universal/data/mnist/mnist2500_X.txt")))
(defonce mnist-labels (str/split-lines (slurp "https://raw.githubusercontent.com/haifengl/smile/refs/heads/master/shell/src/universal/data/mnist/mnist2500_labels.txt")))



(def tsne-2 (TSNE. (-> smile-mnist tc/dataset (tc/rows :as-double-arrays)) 2 5 200 1000))

(plotly/layer-point (-> (. tsne-2 coordinates)
                        (tc/dataset)
                        (tc/add-column :label mnist-labels))
 {:=width 800
  :=x 0
  :=y 1
  :=color :label})

(def vecs
  (->> (json/read-value (.serializeToJson db-store)
                        json/keyword-keys-object-mapper)
       :entries
       (mapv (fn [e] (:vector (:embedding e))))
       (tc/dataset)))


(def tsne-3 (TSNE. (-> vecs
                       (tc/select-rows (range 1000))
                       (tc/rows :as-double-arrays)) 2 2 200 1000))

(plotly/layer-point
 (-> (. tsne-3 coordinates)
     (tc/dataset)
     (tc/add-column :label (take 1000 (-> ds :topic))))
 {:=x 0
  :=y 1
  :=color :label
  :=height 600
  :=width 600})


(def ds-schools-immigration
  (-> ds
      (tc/select-rows #(some #{(:topic %)} ["Schools Building Projects" "International Protection"]))
      (tc/select-columns [:question :topic])
      (tc/map-columns :embedding [:question] (fn [q]
                                               (->> (TextSegment/from q)
                                                    (. model embed)
                                                    (.content)
                                                    (.vector)
                                                    vec)))))

(def tsne-4
  (TSNE.
   (-> (into [] (:embedding ds-schools-immigration))
       (tc/dataset)
       (tc/rows :as-double-arrays))
   2 25 200 1000))


(plotly/layer-point
 (-> (. tsne-4 coordinates)
     (tc/dataset)
     (tc/add-column :label (:topic ds-schools-immigration)))
 {:=x 0
  :=y 1
  :=color :label
  :=height 600
  :=width 600})


;; Filtering for topics that have more than 20 questions associated
(def dep-justice-target-topics
  (-> ds
      (tc/select-rows #(= (:department %) "Justice"))
      (tc/group-by [:topic])
      (tc/aggregate {:n-questions tc/row-count})
      (tc/select-rows #(> (:n-questions %) 20))
      :topic))


(def ds-department-justice
  (-> ds
      (tc/select-rows #(= (:department %) "Justice"))
      (tc/select-rows #(some #{(:topic %)} dep-justice-target-topics))
      (tc/select-columns [:question :topic])
      (tc/map-columns :embedding [:question] (fn [q]
                                               (->> (TextSegment/from q)
                                                    (. model embed)
                                                    (.content)
                                                    (.vector)
                                                    vec)))))


(def tsne-dep-justice
  (TSNE.
   (-> (into [] (:embedding ds-department-justice))
       (tc/dataset)
       (tc/rows :as-double-arrays))
   2 20 200 1000))


(plotly/layer-point
 (-> (. tsne-dep-justice coordinates)
     (tc/dataset)
     (tc/add-column :label (:topic ds-department-justice)))
 {:=x 0
  :=y 1
  :=color :label
  :=height 600
  :=width 600})


(def dep-children-target-topics
  (-> ds
      (tc/select-rows #(= (:department %) "Children"))
      (tc/group-by [:topic])
      (tc/aggregate {:n-questions tc/row-count})
      (tc/select-rows #(> (:n-questions %) 20))
      :topic))


(def ds-department-children
  (-> ds
      (tc/select-rows #(= (:department %) "Children"))
      (tc/select-rows #(some #{(:topic %)} dep-children-target-topics))
      (tc/select-columns [:question :topic])
      (tc/map-columns :embedding [:question] (fn [q]
                                               (->> (TextSegment/from q)
                                                    (. model embed)
                                                    (.content)
                                                    (.vector)
                                                    vec)))))


(def tsne-dep-children
  (TSNE.
   (-> (into [] (:embedding ds-department-children))
       (tc/dataset)
       (tc/rows :as-double-arrays))
   2 20 200 1000))

(-> (. tsne-dep-children coordinates)
    (tc/dataset)
    (tc/add-column :label (:topic ds-department-children))
    (plotly/base
     {:=width 800
      :=height 600})
    (plotly/layer-point
     {:=x 0
      :=y 1
      :=color :label}))


(kind/vega
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :data {:values
         (-> (. tsne-dep-justice coordinates)
             (tc/dataset)
             (tc/add-column :label (:topic ds-department-justice))
             (tc/rename-columns {0 :x 1 :y})
             (tc/rows :as-maps))}
  :width 600
  :height 600
  :mark {:type :point :filled true :size 100}
  :encoding {:x {:field :x
                 :type :quantitative}
             :y {:field :y
                 :type :quantitative}
             :color {:field :label
                     :type :nominal}}})

(kind/vega
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :data {:values
         (-> (. tsne-dep-children coordinates)
             (tc/dataset)
             (tc/add-column :label (:topic ds-department-children))
             (tc/add-column :question (:question ds-department-children))
             (tc/rename-columns {0 :x 1 :y})
             (tc/rows :as-maps))}
  :width 600
  :height 600
  :mark {:type :point :filled true :size 100}
  :encoding {:x {:field :x
                 :type :quantitative}
             :y {:field :y
                 :type :quantitative}
             :color {:field :label
                     :type :nominal}
             :tooltip {:field :question
                       :type :nominal}}})

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :data {:values
         (-> (. tsne-4 coordinates)
             (tc/dataset)
             (tc/add-column :label (:topic ds-schools-immigration))
             (tc/rename-columns {0 :x 1 :y})
             (tc/rows :as-maps))}
  :width 600
  :height 600
  :mark :point
  :encoding {:x {:field :x
                 :type :quantitative}
             :y {:field :y
                 :type :quantitative}
             :color {:field :label
                     :type :nominal}}})


;; Visualise a question and the selected matching questions (neighbours)

(def questions-subset
  (->
   (take 1000 (repeatedly #(rand-nth questions-list)))
   distinct))

(count questions-subset)


(def sample-question "How much has Enterprise Ireland spent on supporting Irish businesses in recent years?")

(def questions-subset-store (InMemoryEmbeddingStore/new))

(count (map #(add-question-to-store! % questions-subset-store) questions-subset))

(def matching-questions
  (let [query (.content (. model embed sample-question))
        matches (. questions-subset-store findRelevant query 5)]
    (map (fn [entry] {:question (.text (.embedded entry))
                      :similarity (.score entry)})
         matches)))

(def ds-subset
  (-> ds
      (tc/select-columns :question)
      (tc/select-rows #(some #{(:question %)} questions-subset))))


(def ds-subset-labelled
  (->> (tc/rows ds-subset :as-maps)
       (map (fn [entry]
              (let [match (filterv (fn [{:keys [question]}]
                                     (= question (:question entry)))
                                   matching-questions)]
                (if (seq match)
                  (-> entry
                      (assoc :label "match")
                      (assoc :similarity (:similarity (first match))))
                  (-> entry
                      (assoc :label "default")
                      (assoc :similarity 0))))))
       (into [{:question sample-question
               :label "question"
               :similarity 0}])
       (tc/dataset)))

(def ds-subset-labelled-embeddings
  (tc/map-columns ds-subset-labelled
                  :embedding [:question]
                  (fn [q]
                    (->> (TextSegment/from q)
                         (. model embed)
                         (.content)
                         (.vector)
                         vec))))

(def tsne-question-asked
  (TSNE.
   (-> (into [] (:embedding ds-subset-labelled-embeddings))
       (tc/dataset)
       (tc/rows :as-double-arrays))
   2 20 200 2000))


(kind/vega
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :data {:values
         (-> (. tsne-question-asked coordinates)
             (tc/dataset)
             (tc/add-column :label (:label ds-subset-labelled))
             (tc/add-column :question (:question ds-subset-labelled))
             (tc/add-column :similarity (:similarity ds-subset-labelled))
             (tc/rename-columns {0 :x 1 :y})
             (tc/rows :as-maps))}
  :width 600
  :height 600
  :mark {:type :point :filled true :size 100}
  :encoding {:x {:field :x
                 :type :quantitative}
             :y {:field :y
                 :type :quantitative}
             :color {:field :label
                     :type :nominal}
             :tooltip {:field :similarity
                       :type :quantitative}}})


(def tsne-question-asked-3d
  (TSNE.
   (-> (into [] (:embedding ds-subset-labelled-embeddings))
       (tc/dataset)
       (tc/rows :as-double-arrays))
   3 20 200 2000))

(-> (. tsne-question-asked-3d coordinates)
    (tc/dataset)
    (tc/add-column :label (:label ds-subset-labelled))
    (plotly/base
     {:=width 800
      :=coordinates :3d})
    (plotly/layer-point
     {:=x 0
      :=y 1
      :=z 2
      :=color :label
      :=mark-opacity 0.7}))
