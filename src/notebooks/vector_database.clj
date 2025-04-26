;; # 2. Vector Database
(ns notebooks.vector-database
  (:require [tablecloth.api :as tc]
            [notebooks.preparation :refer [ds]]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.tableplot.v1.plotly :as plotly]
            [jsonista.core :as json])
  (:import (dev.langchain4j.data.segment TextSegment)
           (dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore)
           (dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel)
           (smile.manifold TSNE)))

;; In this section we will:
;;
;; 1. Build an in-memory vector database using langchain4j
;;
;; 2. Explore visualisations of this database to get an intuition about how the data is stored


;; ## Building a Vector Database from the Questions

;; ### Vector Embeddings
;; First, let's separate out all of the questions from the original dataset, and
;; store these under the name 'questions-list'. We will also take a peek at the
;; first question as a reminder of the type of data we will be storing.

(def questions-list (:question ds))

(kind/pprint (first questions-list))

;; Next, let's see what an vector embedding of this single question looks like.
;; To do this, we have to first define a model that will perform the translation
;; of the text into a vector of numbers. We will use langchain4j's embedding
;; model for this.

(def embedding-model (AllMiniLmL6V2EmbeddingModel/new))

(def sample-question-embedding
  (->> (TextSegment/from (first questions-list))
       (. embedding-model embed)))


(kind/pprint (-> sample-question-embedding
                 .content
                 .vector))

;; We can see it is just an array of floating point numbers. Let's see how large
;; the vector embedding is:

(-> sample-question-embedding
    .content
    .vector
    count)

;; In order to store the embeddings in a database, we will use an in-memory database provided
;; by langchain4j. There are also options for using a more robust solution like postgres, but
;; for testing/exploration purposes, an in-memory database will do.

;; We will first define a short function for adding a question to a memory store:

(defn add-question-to-store! [question store]
  (let [segment (TextSegment/from question)
        embedding (->> segment (. embedding-model embed) (.content))]
    (. store add embedding segment)))

;; At this stage, we would define a new store and add the questions. The
;; in-memory database store function provided by langchain4j also contains an
;; option to convert the data to json. For performance/convenience reasons, I have
;; already pre-made that json file and the `db-store-questions` variable below simply reads
;; from that file to load the database into memory. I have left the code used to
;; generate the json file in as a comment for reference. You can evaluate the
;; code within this comment if you want to re-build the database.

(comment
  (let [db-store (InMemoryEmbeddingStore/new)
        entries-added (count (map #(add-question-to-store! % db-store) questions-list))]
    (do
      (spit "data/retrieval_store/db-store-questions.json" (.serializeToJson db-store))
      (println (str entries-added " records serialised to a json file at data/db-store-questions.json")))))

(def db-store-questions (InMemoryEmbeddingStore/fromFile "data/retrieval_store/db-store-questions.json"))


;; ### Testing Question Lookup

;; This function takes in some 'text' as a query, and returns 'n' number of
;; similar questions, along with their similarity score.

(defn query-db-store [text n]
  (let [query (.content (. embedding-model embed text))
        result (. db-store-questions findRelevant query n)]
    (map (fn [entry]
           {:text (.text (.embedded entry))
            :score (.score entry)})
         result)))

;; As a test question, I'll take an actual question that was asked at a point in
;; time later than that captured by the dataset. It's a question about electric
;; vehicle charging points at airports.

(def test-question-1 "Deputy Emer Currie asked the Minister for Transport his plans to expand EV charging points at the State's airports to facilitate more EV drivers and an increase in EV car rental; and if he will make a statement on the matter.")

(kind/md test-question-1)

;; Now, let's see what questions are similar.

(kind/table
 (query-db-store test-question-1 5))

;; As we can see, there appear to be no pre-existing questions within the
;; timeframe of the dataset that relate specifically to EV charging at
;; *airports*. However, we were able to retrieve questions that generally
;; related to EV charging.

;; Let's try with a question that is similar to something already in the database.

(-> ds
    (tc/select-rows #(= (% :topic) "Passport Services"))
    :question
    (nth 10)
    (kind/md))


(def test-question-2 "The Deputy asked the Minister for Foreign Affairs if he can put in place an effective process for applications for passport renewals or new passport applications")

(kind/table
 (query-db-store test-question-2 5))

;; We can see that we do indeed return the matching question, along with other
;; questions relating to individuals who are experiencing delay's with their
;; passport applications.


;; ### Answering a Question
;;
;; Let's finally use this method to return the best answers based on the question given. This
;; approach could be later used to provide context for a RAG model that would generate answers.

(defn get-answers-for-question [question]
  (let [matching-questions (map :text (query-db-store question 5))]
    (-> ds
        (tc/select-rows #(some #{(% :question)} matching-questions))
        (tc/select-columns [:answer :date]))))

;; As we can see in the table below, we also capture the date the question was
;; answered, and this could be useful in the context of a RAG model when we
;; might perhaps want to return a less similar, but more recent answer.

(get-answers-for-question test-question-2)


;; ## Visualising Question Similarity
;;
;; Now that we have a database that can be used to retrieve similar questions,
;; let's look at a few ways that we can better understand how this process is
;; working.

;; ### Question Similarity Curves

;; For this section, we are going to compare a question against all other
;; questions in the same group. The sample 'groups' we will use are based on
;; three of the most common topics in the dataset:
;;
;; - Schools Building Projects
;;
;; - International Protection
;;
;; - Health Services



(def db-store-schools (InMemoryEmbeddingStore/new))
(def db-store-ip (InMemoryEmbeddingStore/new))
(def db-store-health (InMemoryEmbeddingStore/new))

;; This next step adds the relevant entrie to the above databases. We will reuse
;; our 'add-question-to-store!' function from earlier.

(let [schools-questions (:question (tc/select-rows ds #(= (:topic %) "Schools Building Projects")))
      ip-questions (:question (tc/select-rows ds #(= (:topic %) "International Protection")))
      health-questions (:question (tc/select-rows ds #(= (:topic %) "Health Services")))]
  (do
    (count (mapv #(add-question-to-store! % db-store-schools) schools-questions))
    (count (mapv #(add-question-to-store! % db-store-ip) ip-questions))
    (count (mapv #(add-question-to-store! % db-store-health) health-questions))))


;; Next, we will write a function that, given a question, checks its similarity
;; across all the other questions in a store.

(defn db-store-size
  "A temp/roundabout want to count the size of db
  TODO: find better method"
  [store]
  (-> (json/read-value (.serializeToJson store) json/keyword-keys-object-mapper)
      :entries
      count))

(defn all-similarity-scores [question store & ignore-self?]
  (let [emb (.content (. embedding-model embed question))
        result (. store findRelevant emb (db-store-size store))]
    (map-indexed (fn [idx e] {:idx idx :score (.score e)})
                 (if ignore-self? (rest result) result))))

(def test-question-3 (first (:question (tc/select-rows ds #(= (:topic %) "International Protection")))))


;; TODO: maybe cdf graph instead
;;
;; Below, we simply check a question against all the other questions in the
;; store (excluding itself) and plot all the similarity scores.
;;
(-> (all-similarity-scores test-question-3 db-store-ip true)
    (tc/dataset)
    (plotly/layer-line
     {:=x :idx
      :=y :score}))


;; Let's do the same thing with multiple random questions.

(defn plot-n-random-scores [topic store n plot-type]
  (let [candidates (:question (tc/select-rows ds #(= (:topic %) topic)))
        selected (take n (repeatedly #(rand-nth candidates)))
        data (loop [n 0 result []]
               (if (= n (count selected)) result
                   (let [q (nth selected n)
                         scores (all-similarity-scores q store true)
                         scores (map #(assoc % :question n) scores)]
                     (recur (inc n) (into result scores)))))]
    (if (= plot-type :hist)
      (-> (tc/dataset data)
          (plotly/layer-histogram
           {:=x :score}))
      (-> (tc/dataset data)
          (plotly/base
           {:=x :idx
            :=y :score})
          (plotly/layer-point)
          (plotly/layer-smooth)))))

(plot-n-random-scores "International Protection" db-store-ip 25 :line)

(plot-n-random-scores "Schools Building Projects" db-store-schools 25 :line)

(plot-n-random-scores "Health Services" db-store-health 25 :line)

;; We can also plot these as a histogram:


(plot-n-random-scores "International Protection" db-store-ip 25 :hist)

(plot-n-random-scores "Schools Building Projects" db-store-schools 25 :hist)

(plot-n-random-scores "Health Services" db-store-health 25 :hist)


;; For comparison, let's use a "Health Services" question and compare how
;; similar it is across the three sets.

(def test-health-question (last (:question (tc/select-rows ds #(= (:topic %) "Health Services")))))


(def scores-self-group-other-group
  (let [scores-health (all-similarity-scores test-health-question db-store-health true)
        scores-health (map (fn [s] (assoc s :batch "Health Services")) scores-health)
        scores-sch (all-similarity-scores test-health-question db-store-schools)
        scores-sch (map (fn [s] (assoc s :batch "Schools Building Projects")) scores-sch)
        scores-ip (all-similarity-scores test-health-question db-store-ip true)
        scores-ip (map (fn [s] (assoc s :batch "International Protection")) scores-ip)]
    (-> (concat scores-ip scores-sch scores-health)
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

;; As we can see, in general the question is more similar to other questions in
;; its own topic area than the questions from other topic areas.

;; ## Visualising with tSNE
;;
;; As we saw above, our questions are embedded in the database as a vector of
;; numbers. In a certain sense, we can think of these numbers as coordinates
;; within a high-dimensional space.
;;
;; What the 't-distributed stochastic neighbor embedding' (t-SNE) method allows
;; us to do is reduce these high-dimensional vectors into a 2d or 3d set of
;; coordinates.
;;
;; A nice analogy of what is happening here is when the 3d/spherical space of
;; the Earth's surface is translated into a 2d map. The importance of this
;; analogy lies in the caveats that accopany these 2d representations of the
;; earth. Different projections can convey different significations, a point
;; that his highlighted in a scene from the West Wing where the "Cartophraphers
;; for social equality" give a presentation to the staff on the socio-political
;; implications of different projection techniques.
;; - https://www.youtube.com/watch?v=AMfXVWFBrVo
;;
;; Similarly, with t-SNE and related methods, we have to remain conscious of the
;; limitations of this kind of approach. There is a nice [blog
;; post](https://distill.pub/2016/misread-tsne/) about some of the pitfalls
;; possible when creating t-SNE plots, as well as [podcast
;; episode](https://podcasts.apple.com/us/podcast/the-umap-algorithm-look-pretty-and-do-as-little-as-possible/id1608086450?i=1000679885413)
;; discussing the limitation of UMAP, a similar dimension-reduction method.
;;
;; In the case below, we are using this technique to help illustrate the
;; relative proximity of questions in the database. For example, in the above
;; cases where we find matching questions, are these matches 'neighbours' in a
;; spatial sense? But as with the above caveats, we shouldn't read too much into
;; things like density or clusters or the magnitute of distance. At the same
;; time might help us with our intution around how 'similarity' works across the
;; dataset. It is also interesting to explore what kind of tools we can use
;; within clojure!
;;
;; We will first look at wheather 'topic' areas appear related within this
;; coordinate space, and then will try map where a sample question and its
;; matches lie.

;; ### Topic Areas Projection

;; We will take the three topic areas we already used above:
;;
;; - Schools Building Projects
;;
;; - International Protection
;;
;; - Health Services

(def ds-schools-immigration-health
  (-> ds
      (tc/select-rows #(some #{(:topic %)} ["Schools Building Projects" "International Protection" "Health Services"]))
      (tc/select-columns [:question :topic])
      (tc/map-columns :embedding [:question] (fn [q]
                                               (->> (TextSegment/from q)
                                                    (. embedding-model embed)
                                                    (.content)
                                                    (.vector)
                                                    vec)))))

;; To reduce the embedding to a 2d space, we will use the Smile implementation of t-SNE.
;;
;; This function takes in 4 parameters as arguments:
;;
;; - number of dimensions
;;
;; - perplexity
;;
;; - learning rate
;;
;; - iterations
;;
;; The two most impactful variables are 'perplexity' and 'iterations'.
;; Perplexity is something like the expected group 'size'. For example, if you
;; had a dataset with a hundred points and a perpexity of 100, the alorithm
;; would try to keep these points close together in the coordinate space.
;;
;; We also need to use tablecloth's transformation of the data to 'double-arrays'
;; to transform the data to a datatype expected by the model.
;;

(defn make-t-sne-coords [dataset {:keys [dimensions perplexity learning-rate iterations]
                                  :or {dimensions 2
                                       perplexity 25
                                       learning-rate 200
                                       iterations 1000}}]
  (let [ds (-> (into [] (:embedding dataset))
               (tc/dataset)
               (tc/rows :as-double-arrays)
               (TSNE. dimensions perplexity learning-rate iterations)
               (. coordinates)
               (tc/dataset))]
    (if (= dimensions 2)
      (tc/rename-columns ds [:x :y])
      (tc/rename-columns ds [:x :y :z]))))

(defn plot-t-sne-coords [ds labels t-sne-opts plot-opts]
  (-> ds
      (make-t-sne-coords t-sne-opts)
      (tc/add-column :label labels)
      (plotly/base {:=width 700})
      (plotly/layer-point plot-opts)))

(plot-t-sne-coords ds-schools-immigration-health
                   (:topic ds-schools-immigration-health)
                   {:iterations 100}
                   {:=color :label})

(plot-t-sne-coords ds-schools-immigration-health
                   (:topic ds-schools-immigration-health)
                   {:iterations 500}
                   {:=color :label})

(plot-t-sne-coords ds-schools-immigration-health
                   (:topic ds-schools-immigration-health)
                   {:iterations 1000}
                   {:=color :label})

(plot-t-sne-coords ds-schools-immigration-health
                   (:topic ds-schools-immigration-health)
                   {:iterations 1000
                    :perplexity 3}
                   {:=color :label})

(plot-t-sne-coords ds-schools-immigration-health
                   (:topic ds-schools-immigration-health)
                   {:iterations 1000
                    :perplexity 300}
                   {:=color :label})


;; After 100 iterations (quite early in the process), there isn't much
;; separation at all. After 500 iterations, the points begin to split apart.
;; Wildly varying the perplexity doesn't seem to have a huge imapact on this
;; visualisation.



;; ### Ploting a Department's question-set

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
                                                    (. embedding-model embed)
                                                    (.content)
                                                    (.vector)
                                                    vec)))))

(plot-t-sne-coords ds-department-justice
                   (:topic ds-department-justice)
                   {:perplexity 10
                    :iterations 5000}
                   {:=color :label})


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
                                                    (. embedding-model embed)
                                                    (.content)
                                                    (.vector)
                                                    vec)))))

(plot-t-sne-coords ds-department-children
                   (:topic ds-department-children)
                   {:perplexity 10
                    :iterations 2000}
                   {:=color :label})





;; ### Visualise a question and the selected matching questions (neighbours)

(def ds-subset
  (-> ds
      (tc/select-rows (range 1000))))

(-> ds-subset
    (tc/group-by [:topic])
    (tc/aggregate tc/row-count)
    (tc/order-by "summary" :desc)
    (tc/select-rows (range 5)))

;; Of these first 1,000 questions, it seems like there are 25 questions relating
;; to "Housing Schemes", so our test question will be along those lines.

(def sample-question "The Deputy asks the Minister a question about the number of vacant houses")

(def questions-subset-store (InMemoryEmbeddingStore/new))

(count (map #(add-question-to-store! % questions-subset-store) (:question ds-subset)))

;; This function, similar to ones above, returns the 5 closest matching
;; questions. These will be what we try to visualise.

(def matching-questions
  (let [query (.content (. embedding-model embed sample-question))
        matches (. questions-subset-store findRelevant query 5)]
    (map (fn [entry] {:question (.text (.embedded entry))
                      :similarity (.score entry)})
         matches)))

;; Next, we will add custom labels to the data. We will also add our sample
;; question into the dataset and label it accordingly. The labels will be:
;;
;; - 'default' - the quesitons that haven't been matched
;; - 'match' - one of the 5 matching questions
;; - 'question' - the question itself

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
               :similarity 1}])
       (tc/dataset)))


;; Finally, we will add the embeddings to the questions.

;; TODO: this fn to add embeddings is re-used a lot. Refactor
(def ds-subset-labelled-embeddings
  (tc/map-columns ds-subset-labelled
                  :embedding [:question]
                  (fn [q]
                    (->> (TextSegment/from q)
                         (. embedding-model embed)
                         (.content)
                         (.vector)
                         vec))))



(plot-t-sne-coords ds-subset-labelled-embeddings
                   (:label ds-subset-labelled-embeddings)
                   {}
                   {:=color :label
                    :=mark-opacity 0.6})


(def q-test-ds
  (-> (make-t-sne-coords ds-subset-labelled-embeddings {:perplexity 10})
      (tc/dataset)
      (tc/rename-columns [:x :y])
      (tc/add-column :label (:label ds-subset-labelled))
      (tc/add-column :question (:question ds-subset-labelled))
      (tc/add-column :similarity (:similarity ds-subset-labelled))))

(kind/echarts
 {:tooltip {}
  :xAxis {}
  :yAxis {}
  :series [{:data (-> q-test-ds
                      (tc/select-rows #(= (% :label) "question"))
                      (tc/select-columns [:x :y])
                      (tc/rows :as-vectors))
            :name "Question"
            :symbolSize 20
            :type "scatter"}
           {:data (-> q-test-ds
                      (tc/select-rows #(= (% :label) "match"))
                      (tc/select-columns [:x :y])
                      (tc/rows :as-vectors))
            :name "Match"
            :type "scatter"
            :symbolSize 15}
           {:data (-> q-test-ds
                      (tc/select-rows #(= (% :label) "default"))
                      (tc/select-columns [:x :y])
                      (tc/rows :as-vectors))
            :itemStyle {:opacity 0.3}
            :name "Default"
            :type "scatter"}]})


;; 3d Plot

(plot-t-sne-coords ds-subset-labelled-embeddings
                   (:label ds-subset-labelled-embeddings)
                   {:perplexity 10 :dimensions 3}
                   {:=color :label :=coordinates :3d
                    :=mark-opacity 0.5})
