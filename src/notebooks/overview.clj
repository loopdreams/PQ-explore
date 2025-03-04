;; # Dataset Overview
(ns notebooks.overview
  (:require
   [clojure.string :as str]
   [java-time.api :as jt]
   [notebooks.preparation :refer [ds]]
   [scicloj.kindly.v4.kind :as kind]
   [tablecloth.api :as tc]
   [scicloj.tableplot.v1.plotly :as plotly]))

;; ## General Stats

^:kindly/hide-code
(def total-questions (tc/row-count ds))

^:kindly/hide-code
(def number-deputies (-> ds :member distinct count))

^:kindly/hide-code
(def average-questions-annual (float (/ total-questions
                                        (- (-> ds
                                               (tc/map-columns :year [:date] jt/year)
                                               :year distinct count)
                                           1))))

^:kindly/hide-code
(def average-questions-monthly (let [m-counts (-> ds
                                                  (tc/map-columns :year-month [:date] jt/year-month)
                                                  (tc/group-by [:year-month])
                                                  (tc/aggregate {:count tc/row-count})
                                                  :count)]
                                 (float (/ (reduce + m-counts) (count m-counts)))))

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
   [:li [:strong (format "%,d" total-questions)] " total questions asked by " [:strong number-deputies] " members of parliament"]
   [:li [:strong (format "%,.0f" average-questions-annual)] " questions asked on average annually"]
   [:li [:strong (format "%,.0f" average-questions-monthly)] " questions asked on average monthly"]
   [:li [:strong (format "%,d" number-topics)] " distinct question topics"]
   [:li "The five most common question topics are: " (str/join ", " top-5-topics)]
   [:li "The five most commonnly asked departments are: " (str/join ", " top-5-most-asked-departments)]]])

;; ## Departments/Topics Visual

;; Since there are so many topics across the dataset, let's first try to narrow it down
;; by year (2024) and to only topics that have a large number of questions associated
;;
;; To try see what constitutes a 'large' number of questions, let's make a histogram based
;; on how many questions are asked in a topic area.

(def ds-grouped-topic-counts
  (-> ds
      ;; (tc/select-rows #(some #{(% :department)} ["Health" "Justice" "Environment" "Children" "Finance"]))
      (tc/select-rows #(some #{(str (jt/year (% :date)))} ["2024"]))
      (tc/group-by [:topic])
      (tc/aggregate {:n-questions tc/row-count})))


(-> ds-grouped-topic-counts
    (plotly/layer-histogram
     {:=x :n-questions
      :=histnorm "questions"
      :=histogram-nbins 10}))


;; Not a very pretty histogram! But it does tell us that the vast majority of topics
;; had in 2024 less than 50 questions.
;;
;; Let's drop some of those low-question topics and see how many topics are left.

(-> ds-grouped-topic-counts
    (tc/drop-rows #(< (% :n-questions) 200))
    (tc/row-count))

(def target-topics
  (into #{}
        (-> ds-grouped-topic-counts
            (tc/drop-rows #(< (% :n-questions) 200))
            :topic)))

(def ds-target-topics
  (-> ds
      (tc/select-rows #(some #{(str (jt/year (% :date)))} ["2024"]))
      (tc/select-rows #(target-topics (% :topic)))))

;; Next we need to re-process the data into a format like this:
;; {:name "name" :children [:name "child-name" :value x]}

(defn department-topic-format [ds]
  (let [topics-frequencies (-> ds :topic frequencies)
        topics-map (reduce (fn [res [topic count]]
                             (if (< count 50) res
                                 (conj res
                                       {:name topic
                                        :value count})))
                           []
                           topics-frequencies)
        parent (-> ds :department first)]
    {:name parent
     :children topics-map}))



(def treemap-data-format
  (let [datasets (-> ds-target-topics
                     (tc/group-by :department)
                     :data)
        all-children (mapv department-topic-format datasets)]
    {:name ""
     :children all-children}))


(kind/echarts
 {:tooltip {:trigger "item"
            :triggerOn "mousemove"}
  :series
  [{:type "tree"
    :data [treemap-data-format]
    :top "18%"
    :bottom "14%"
    :layout "radial"
    :symbol "emptyCircle"
    :symbolSize 7
    :initialTreeDepth 3
    :animationDurationUpdate 750
    :emphasis {:focus "descendant"}}]})

;; Sankey
;;
(def sankey-data-format
  (let [ds-t ds-target-topics
        nodes (mapv (fn [name] {:name name}) (-> ds-t :department distinct))
        nodes (into nodes (mapv (fn [name] {:name name}) (-> ds-t :topic distinct)))
        links (reduce (fn [res ds]
                        (let [source (-> ds :department first)
                              values (-> ds :topic frequencies)]
                          (into res
                                (mapv (fn [[name value]] {:source source :target name :value value}) values))))
                      []
                      (-> ds-t (tc/group-by :department) :data))]
    {:nodes nodes
     :links links}))

;; Not sure how to adjust size here..
(kind/echarts
 ;; {:style {:width "800px" :height "600px"}
 ;;  :option
  {:toolip {:trigger "item"
            :triggerOn "mousemove"}
   :series [{:type "sankey"
             :data (:nodes sankey-data-format)
             :links (:links sankey-data-format)
             :emphasis {:focus "adjacency"}}]})

;; If you hover over the chart you can see some topics span multiple departments.
;; Let's have a quick look at some of those topics

(defn topic-sankey-data [ds topic]
  (let [ds    (tc/select-rows ds #(and (= (str (jt/year (% :date))) "2024")
                                       (= (% :topic) topic)))
        nodes (conj (mapv (fn [name] {:name name}) (-> ds :department distinct))
                    {:name topic})
        links (reduce (fn [res ds]
                        (let [target (-> ds :department first)
                              value  (tc/row-count ds)]
                          (conj res {:source topic :target target :value value})))
                      []
                      (-> ds (tc/group-by :department) :data))]
    {:nodes nodes
     :links links}))

(defn topic-sankey [ds topic]
  (let [{:keys [nodes links]} (topic-sankey-data ds topic)]
    (kind/echarts
       {:toolip {:trigger "item"
                 :triggerOn "mousemove"}
        :series [{:type "sankey"
                  :data nodes
                  :links links
                  :emphasis {:focus "adjacency"}}]})))

(topic-sankey ds "Disability Services")

(topic-sankey ds "An Garda Síochána")

(topic-sankey ds "Departmental Staff")

(topic-sankey ds "Business Supports")
