# [Remaining Useful Life Prediction Papers](https://ramtinmoslemi.github.io/RUL-Papers/)
A complete list of papers on **Remaining Useful Life (RUL) Prediction**, **State of Health (SOH) Prediction**, and **Predictive Maintenance (PdM)** submitted to arXiv over the past decade.

You can find the papers and their titles, abstracts, authors, links, and dates stored in [this csv file](https://github.com/RamtinMoslemi/RUL-Papers/blob/main/rul_papers.csv).

## Paper Counts by Year
Number of papers submitted to arXiv by year.

![yearly_papers](figures/paper_by_year.svg)

## Word Clouds
Word clouds of paper titles and abstracts.

![word_clouds](figures/word_clouds.png)

## Notebook
You can play with the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RamtinMoslemi/RUL-Papers/blob/main/RUL_Papers.ipynb) [![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/RamtinMoslemi/RUL-Papers/main/RUL_Papers.ipynb)


# 2014
## May
### [Spreading of diseases through comorbidity networks across life and gender](https://arxiv.org/abs/1405.3801)

**Authors:**
Anna Chmiel, Peter Klimek, Stefan Thurner

**Abstract:**
The state of health of patients is typically not characterized by a single disease alone but by multiple (comorbid) medical conditions. These comorbidities may depend strongly on age and gender. We propose a specific phenomenological comorbidity network of human diseases that is based on medical claims data of the entire population of Austria. The network is constructed from a two-layer multiplex network, where in one layer the links represent the conditional probability for a comorbidity, and in the other the links contain the respective statistical significance. We show that the network undergoes dramatic structural changes across the lifetime of patients.Disease networks for children consist of a single, strongly inter-connected cluster. During adolescence and adulthood further disease clusters emerge that are related to specific classes of diseases, such as circulatory, mental, or genitourinary disorders.For people above 65 these clusters start to merge and highly connected hubs dominate the network. These hubs are related to hypertension, chronic ischemic heart diseases, and chronic obstructive pulmonary diseases. We introduce a simple diffusion model to understand the spreading of diseases on the disease network at the population level. For the first time we are able to show that patients predominantly develop diseases which are in close network-proximity to disorders that they already suffer. The model explains more than 85 % of the variance of all disease incidents in the population. The presented methodology could be of importance for anticipating age-dependent disease-profiles for entire populations, and for validation and of prevention schemes.
       


### [Using the Expectation Maximization Algorithm with Heterogeneous Mixture Components for the Analysis of Spectrometry Data](https://arxiv.org/abs/1405.5501)

**Authors:**
Dominik Kopczynski, Sven Rahmann

**Abstract:**
Coupling a multi-capillary column (MCC) with an ion mobility (IM) spectrometer (IMS) opened a multitude of new application areas for gas analysis, especially in a medical context, as volatile organic compounds (VOCs) in exhaled breath can hint at a person's state of health. To obtain a potential diagnosis from a raw MCC/IMS measurement, several computational steps are necessary, which so far have required manual interaction, e.g., human evaluation of discovered peaks. We have recently proposed an automated pipeline for this task that does not require human intervention during the analysis. Nevertheless, there is a need for improved methods for each computational step. In comparison to gas chromatography / mass spectrometry (GC/MS) data, MCC/IMS data is easier and less expensive to obtain, but peaks are more diffuse and there is a higher noise level. MCC/IMS measurements can be described as samples of mixture models (i.e., of convex combinations) of two-dimensional probability distributions. So we use the expectation-maximization (EM) algorithm to deconvolute mixtures in order to develop methods that improve data processing in three computational steps: denoising, baseline correction and peak clustering. A common theme of these methods is that mixture components within one model are not homogeneous (e.g., all Gaussian), but of different types. Evaluation shows that the novel methods outperform the existing ones. We provide Python software implementing all three methods and make our evaluation data available at http://www.rahmannlab.de/research/ims.
       


## December
### [Context-Aware Analytics in MOM Applications](https://arxiv.org/abs/1412.7968)

**Authors:**
Martin Ringsquandl, Steffen Lamparter, Raffaello Lepratti

**Abstract:**
Manufacturing Operations Management (MOM) systems are complex in the sense that they integrate data from heterogeneous systems inside the automation pyramid. The need for context-aware analytics arises from the dynamics of these systems that influence data generation and hamper comparability of analytics, especially predictive models (e.g. predictive maintenance), where concept drift affects application of these models in the future. Recently, an increasing amount of research has been directed towards data integration using semantic context models. Manual construction of such context models is an elaborate and error-prone task. Therefore, we pose the challenge to apply combinations of knowledge extraction techniques in the domain of analytics in MOM, which comprises the scope of data integration within Product Life-cycle Management (PLM), Enterprise Resource Planning (ERP), and Manufacturing Execution Systems (MES). We describe motivations, technological challenges and show benefits of context-aware analytics, which leverage from and regard the interconnectedness of semantic context data. Our example scenario shows the need for distribution and effective change tracking of context information.
       


# 2015
## February
### [Software that Learns from its Own Failures](https://arxiv.org/abs/1502.00821)

**Author:**
Martin Monperrus

**Abstract:**
All non-trivial software systems suffer from unanticipated production failures. However, those systems are passive with respect to failures and do not take advantage of them in order to improve their future behavior: they simply wait for them to happen and trigger hard-coded failure recovery strategies. Instead, I propose a new paradigm in which software systems learn from their own failures. By using an advanced monitoring system they have a constant awareness of their own state and health. They are designed in order to automatically explore alternative recovery strategies inferred from past successful and failed executions. Their recovery capabilities are assessed by self-injection of controlled failures; this process produces knowledge in prevision of future unanticipated failures.
       


### [Towards zero-configuration condition monitoring based on dictionary learning](https://arxiv.org/abs/1502.03596)

**Authors:**
Sergio Martin-del-Campo, Fredrik Sandin

**Abstract:**
Condition-based predictive maintenance can significantly improve overall equipment effectiveness provided that appropriate monitoring methods are used. Online condition monitoring systems are customized to each type of machine and need to be reconfigured when conditions change, which is costly and requires expert knowledge. Basic feature extraction methods limited to signal distribution functions and spectra are commonly used, making it difficult to automatically analyze and compare machine conditions. In this paper, we investigate the possibility to automate the condition monitoring process by continuously learning a dictionary of optimized shift-invariant feature vectors using a well-known sparse approximation method. We study how the feature vectors learned from a vibration signal evolve over time when a fault develops within a ball bearing of a rotating machine. We quantify the adaptation rate of learned features and find that this quantity changes significantly in the transitions between normal and faulty states of operation of the ball bearing.
       


## April
### [Discriminative Switching Linear Dynamical Systems applied to Physiological Condition Monitoring](https://arxiv.org/abs/1504.06494)

**Authors:**
Konstantinos Georgatzis, Christopher K. I. Williams

**Abstract:**
We present a Discriminative Switching Linear Dynamical System (DSLDS) applied to patient monitoring in Intensive Care Units (ICUs). Our approach is based on identifying the state-of-health of a patient given their observed vital signs using a discriminative classifier, and then inferring their underlying physiological values conditioned on this status. The work builds on the Factorial Switching Linear Dynamical System (FSLDS) (Quinn et al., 2009) which has been previously used in a similar setting. The FSLDS is a generative model, whereas the DSLDS is a discriminative model. We demonstrate on two real-world datasets that the DSLDS is able to outperform the FSLDS in most cases of interest, and that an $α$-mixture of the two models achieves higher performance than either of the two models separately.
       


## July
### [A Study of the Management of Electronic Medical Records in Fijian Hospitals](https://arxiv.org/abs/1507.03659)

**Authors:**
Swaran S. Ravindra, Rohitash Chandra, Virallikattur S. Dhenesh

**Abstract:**
Despite having a number of benefits for healthcare settings, the successful implementation of health information systems (HIS) continues to be a challenge in many developing countries. This paper examines the current state of health information systems in government hospitals in Fiji. It also investigates if the general public as well as medical practitioners in Fiji have interest in having web based electronic medical records systems that allow patients to access their medical reports and make online bookings for their appointments. Nausori Health Centre was used as a case study to examine the information systems in a government hospital in Fiji.
       


## November
### [Modeling And Control Battery Aging in Energy Harvesting Systems](https://arxiv.org/abs/1511.03495)

**Authors:**
Roberto Valentini, Nga Dang, Marco Levorato, Eli Bozorgzadeh

**Abstract:**
Energy storage is a fundamental component for the development of sustainable and environment-aware technologies. One of the critical challenges that needs to be overcome is preserving the State of Health (SoH) in energy harvesting systems, where bursty arrival of energy and load may severely degrade the battery. Tools from Markov process and Dynamic Programming theory are becoming an increasingly popular choice to control dynamics of these systems due to their ability to seamlessly incorporate heterogeneous components and support a wide range of applications. Mapping aging rate measures to fit within the boundaries of these tools is non-trivial. In this paper, a framework for modeling and controlling the aging rate of batteries based on Markov process theory is presented. Numerical results illustrate the tradeoff between battery degradation and task completion delay enabled by the proposed framework.
       


# 2016
## February
### [Composable Industrial Internet Applications for Tiered Architectures](https://arxiv.org/abs/1602.05163)

**Authors:**
K. Eric Harper, Thijmen de Gooijer, Karen Smiley

**Abstract:**
A single vendor cannot provide complete IIoT end-to-end solutions because cooperation is required from multiple parties. Interoperability is a key architectural quality. Composability of capabilities, information and configuration is the prerequisite for interoperability, supported by a data storage infrastructure and defined set of interfaces to build applications. Secure collection, transport and storage of data and algorithms are expectations for collaborative participation in any IIoT solution. Participants require control of their data ownership and confidentiality. We propose an Internet of Things, Services and People (IoTSP) application development and management framework which includes components for data storage, algorithm design and packaging, and computation execution. Applications use clusters of platform services, organized in tiers, and local access to data to reduce complexity and enhance reliable data exchange. Since communication is less reliable across tiers, data is synchronized between storage replicas when communication is available. The platform services provide a common ecosystem to exchange data uniting data storage, applications, and components that process the data. Configuration and orchestration of the tiers are managed using shared tools and facilities. The platform promotes the data storage components to be peers of the applications where each data owner is in control of when and how much information is shared with a service provider. The service components and applications are securely integrated using local event and data exchange communication channels. This tiered architecture reduces the cyber attack surface and enables individual tiers to operate autonomously, while addressing interoperability concerns. We present our framework using predictive maintenance as an example, and evaluate compatibility of our vision with an emerging set of standards.
       


## June
### [De-identification of Patient Notes with Recurrent Neural Networks](https://arxiv.org/abs/1606.03475)

**Authors:**
Franck Dernoncourt, Ji Young Lee, Ozlem Uzuner, Peter Szolovits

**Abstract:**
Objective: Patient notes in electronic health records (EHRs) may contain critical information for medical investigations. However, the vast majority of medical investigators can only access de-identified notes, in order to protect the confidentiality of patients. In the United States, the Health Insurance Portability and Accountability Act (HIPAA) defines 18 types of protected health information (PHI) that needs to be removed to de-identify patient notes. Manual de-identification is impractical given the size of EHR databases, the limited number of researchers with access to the non-de-identified notes, and the frequent mistakes of human annotators. A reliable automated de-identification system would consequently be of high value.
  Materials and Methods: We introduce the first de-identification system based on artificial neural networks (ANNs), which requires no handcrafted features or rules, unlike existing systems. We compare the performance of the system with state-of-the-art systems on two datasets: the i2b2 2014 de-identification challenge dataset, which is the largest publicly available de-identification dataset, and the MIMIC de-identification dataset, which we assembled and is twice as large as the i2b2 2014 dataset.
  Results: Our ANN model outperforms the state-of-the-art systems. It yields an F1-score of 97.85 on the i2b2 2014 dataset, with a recall 97.38 and a precision of 97.32, and an F1-score of 99.23 on the MIMIC de-identification dataset, with a recall 99.25 and a precision of 99.06.
  Conclusion: Our findings support the use of ANNs for de-identification of patient notes, as they show better performance than previously published systems while requiring no feature engineering.
       


## August
### [Resiliency in Distributed Sensor Networks for PHM of the Monitoring Targets](https://arxiv.org/abs/1608.05844)

**Authors:**
Jacques Bahi, Wiem Elghazel, Christophe Guyeux, Mohammed Haddad, Mourad Hakem, Kamal Medjaher, Nourredine Zerhouni

**Abstract:**
In condition-based maintenance, real-time observations are crucial for on-line health assessment. When the monitoring system is a wireless sensor network, data loss becomes highly probable and this affects the quality of the remaining useful life prediction. In this paper, we present a fully distributed algorithm that ensures fault tolerance and recovers data loss in wireless sensor networks. We first theoretically analyze the algorithm and give correctness proofs, then provide simulation results and show that the algorithm is (i) able to ensure data recovery with a low failure rate and (ii) preserves the overall energy for dense networks.
       


### [Multi-Sensor Prognostics using an Unsupervised Health Index based on LSTM Encoder-Decoder](https://arxiv.org/abs/1608.06154)

**Authors:**
Pankaj Malhotra, Vishnu TV, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, Gautam Shroff

**Abstract:**
Many approaches for estimation of Remaining Useful Life (RUL) of a machine, using its operational sensor data, make assumptions about how a system degrades or a fault evolves, e.g., exponential degradation. However, in many domains degradation may not follow a pattern. We propose a Long Short Term Memory based Encoder-Decoder (LSTM-ED) scheme to obtain an unsupervised health index (HI) for a system using multi-sensor time-series data. LSTM-ED is trained to reconstruct the time-series corresponding to healthy state of a system. The reconstruction error is used to compute HI which is then used for RUL estimation. We evaluate our approach on publicly available Turbofan Engine and Milling Machine datasets. We also present results on a real-world industry dataset from a pulverizer mill where we find significant correlation between LSTM-ED based HI and maintenance costs.
       


## October
### [Distributed and parallel time series feature extraction for industrial big data applications](https://arxiv.org/abs/1610.07717)

**Authors:**
Maximilian Christ, Andreas W. Kempa-Liehr, Michael Feindt

**Abstract:**
The all-relevant problem of feature selection is the identification of all strongly and weakly relevant attributes. This problem is especially hard to solve for time series classification and regression in industrial applications such as predictive maintenance or production line optimization, for which each label or regression target is associated with several time series and meta-information simultaneously. Here, we are proposing an efficient, scalable feature extraction algorithm for time series, which filters the available features in an early stage of the machine learning pipeline with respect to their significance for the classification or regression task, while controlling the expected percentage of selected but irrelevant features. The proposed algorithm combines established feature extraction methods with a feature importance filter. It has a low computational complexity, allows to start on a problem with only limited domain knowledge available, can be trivially parallelized, is highly scalable and based on well studied non-parametric hypothesis tests. We benchmark our proposed algorithm on all binary classification problems of the UCR time series classification archive as well as time series from a production line optimization project and simulated stochastic processes with underlying qualitative change of dynamics.
       


## November
### [Proposal of Real Time Predictive Maintenance Platform with 3D Printer for Business Vehicles](https://arxiv.org/abs/1611.09944)

**Authors:**
Yoji Yamato, Yoshifumi Fukumoto, Hiroki Kumazaki

**Abstract:**
This paper proposes a maintenance platform for business vehicles which detects failure sign using IoT data on the move, orders to create repair parts by 3D printers and to deliver them to the destination. Recently, IoT and 3D printer technologies have been progressed and application cases to manufacturing and maintenance have been increased. Especially in air flight industry, various sensing data are collected during flight by IoT technologies and parts are created by 3D printers. And IoT platforms which improve development/operation of IoT applications also have been appeared. However, existing IoT platforms mainly targets to visualize "things" statuses by batch processing of collected sensing data, and 3 factors of real-time, automatic orders of repair parts and parts stock cost are insufficient to accelerate businesses. This paper targets maintenance of business vehicles such as airplane or high-speed bus. We propose a maintenance platform with real-time analysis, automatic orders of repair parts and minimum stock cost of parts. The proposed platform collects data via closed VPN, analyzes stream data and predicts failures in real-time by online machine learning framework Jubatus, coordinates ERP or SCM via in memory DB to order repair parts and also distributes repair parts data to 3D printers to create repair parts near the destination.
       


### [Using Temporal and Semantic Developer-Level Information to Predict Maintenance Activity Profiles](https://arxiv.org/abs/1611.10053)

**Authors:**
Stanislav Levin, Amiram Yehudai

**Abstract:**
Predictive models for software projects' characteristics have been traditionally based on project-level metrics, employing only little developer-level information, or none at all. In this work we suggest novel metrics that capture temporal and semantic developer-level information collected on a per developer basis. To address the scalability challenges involved in computing these metrics for each and every developer for a large number of source code repositories, we have built a designated repository mining platform. This platform was used to create a metrics dataset based on processing nearly 1000 highly popular open source GitHub repositories, consisting of 147 million LOC, and maintained by 30,000 developers. The computed metrics were then employed to predict the corrective, perfective, and adaptive maintenance activity profiles identified in previous works. Our results show both strong correlation and promising predictive power with R-squared values of 0.83, 0.64, and 0.75. We also show how these results may help project managers to detect anomalies in the development process and to build better development teams. In addition, the platform we built has the potential to yield further predictive models leveraging developer-level metrics at scale.
       


## December
### [Predicting Patient State-of-Health using Sliding Window and Recurrent Classifiers](https://arxiv.org/abs/1612.00662)

**Authors:**
Adam McCarthy, Christopher K. I. Williams

**Abstract:**
Bedside monitors in Intensive Care Units (ICUs) frequently sound incorrectly, slowing response times and desensitising nurses to alarms (Chambrin, 2001), causing true alarms to be missed (Hug et al., 2011). We compare sliding window predictors with recurrent predictors to classify patient state-of-health from ICU multivariate time series; we report slightly improved performance for the RNN for three out of four targets.
       


### [Realtime Predictive Maintenance with Lambda Architecture](https://arxiv.org/abs/1612.02640)

**Authors:**
Yoji Yamato, Hiroki Kumazaki, Yoshifumi Fukumoto

**Abstract:**
Recently, IoT technologies have been progressed and applications of maintenance area are expected. However, IoT maintenance applications are not spread in Japan yet because of insufficient analysis of real time situation, high cost to collect sensing data and to configure failure detection rules. In this paper, using lambda architecture concept, we propose a maintenance platform in which edge nodes analyze sensing data, detect anomaly, extract a new detection rule in real time and a cloud orders maintenance automatically, also analyzes whole data collected by batch process in detail, updates learning model of edge nodes to improve analysis accuracy.
       


# 2017
## January
### [A dissimilarity-based approach to predictive maintenance with application to HVAC systems](https://arxiv.org/abs/1701.03633)

**Authors:**
Riccardo Satta, Stefano Cavallari, Eraldo Pomponi, Daniele Grasselli, Davide Picheo, Carlo Annis

**Abstract:**
The goal of predictive maintenance is to forecast the occurrence of faults of an appliance, in order to proactively take the necessary actions to ensure its availability. In many application scenarios, predictive maintenance is applied to a set of homogeneous appliances. In this paper, we firstly review taxonomies and main methodologies currently used for condition-based maintenance; secondly, we argue that the mutual dissimilarities of the behaviours of all appliances of this set (the "cohort") can be exploited to detect upcoming faults. Specifically, inspired by dissimilarity-based representations, we propose a novel machine learning approach based on the analysis of concurrent mutual differences of the measurements coming from the cohort. We evaluate our method over one year of historical data from a cohort of 17 HVAC (Heating, Ventilation and Air Conditioning) systems installed in an Italian hospital. We show that certain kinds of faults can be foreseen with an accuracy, measured in terms of area under the ROC curve, as high as 0.96.
       


## March
### [Gaussian process regression for forecasting battery state of health](https://arxiv.org/abs/1703.05687)

**Authors:**
Robert R. Richardson, Michael A. Osborne, David A. Howey

**Abstract:**
Accurately predicting the future capacity and remaining useful life of batteries is necessary to ensure reliable system operation and to minimise maintenance costs. The complex nature of battery degradation has meant that mechanistic modelling of capacity fade has thus far remained intractable; however, with the advent of cloud-connected devices, data from cells in various applications is becoming increasingly available, and the feasibility of data-driven methods for battery prognostics is increasing. Here we propose Gaussian process (GP) regression for forecasting battery state of health, and highlight various advantages of GPs over other data-driven and mechanistic approaches. GPs are a type of Bayesian non-parametric method, and hence can model complex systems whilst handling uncertainty in a principled manner. Prior information can be exploited by GPs in a variety of ways: explicit mean functions can be used if the functional form of the underlying degradation model is available, and multiple-output GPs can effectively exploit correlations between data from different cells. We demonstrate the predictive capability of GPs for short-term and long-term (remaining useful life) forecasting on a selection of capacity vs. cycle datasets from lithium-ion cells.
       


## May
### [Modeling The Intensity Function Of Point Process Via Recurrent Neural Networks](https://arxiv.org/abs/1705.08982)

**Authors:**
Shuai Xiao, Junchi Yan, Stephen M. Chu, Xiaokang Yang, Hongyuan Zha

**Abstract:**
Event sequence, asynchronously generated with random timestamp, is ubiquitous among applications. The precise and arbitrary timestamp can carry important clues about the underlying dynamics, and has lent the event data fundamentally different from the time-series whereby series is indexed with fixed and equal time interval. One expressive mathematical tool for modeling event is point process. The intensity functions of many point processes involve two components: the background and the effect by the history. Due to its inherent spontaneousness, the background can be treated as a time series while the other need to handle the history events. In this paper, we model the background by a Recurrent Neural Network (RNN) with its units aligned with time series indexes while the history effect is modeled by another RNN whose units are aligned with asynchronous events to capture the long-range dynamics. The whole model with event type and timestamp prediction output layers can be trained end-to-end. Our approach takes an RNN perspective to point process, and models its background and history effect. For utility, our method allows a black-box treatment for modeling the intensity which is often a pre-defined parametric form in point processes. Meanwhile end-to-end training opens the venue for reusing existing rich techniques in deep network for point process modeling. We apply our model to the predictive maintenance problem using a log dataset by more than 1000 ATMs from a global bank headquartered in North America.
       


## June
### [Dependability of Sensor Networks for Industrial Prognostics and Health Management](https://arxiv.org/abs/1706.08129)

**Authors:**
Wiem Elghazel, Jacques M. Bahi, Christophe Guyeux, Mourad Hakem, Kamal Medjaher, Noureddine Zerhouni

**Abstract:**
Maintenance is an important activity in industry. It is performed either to revive a machine/component or to prevent it from breaking down. Different strategies have evolved through time, bringing maintenance to its current state: condition-based and predictive maintenances. This evolution was due to the increasing demand of reliability in industry. The key process of condition-based and predictive maintenances is prognostics and health management, and it is a tool to predict the remaining useful life of engineering assets. Nowadays, plants are required to avoid shutdowns while offering safety and reliability. Nevertheless, planning a maintenance activity requires accurate information about the system/component health state. Such information is usually gathered by means of independent sensor nodes. In this study, we consider the case where the nodes are interconnected and form a wireless sensor network. As far as we know, no research work has considered such a case of study for prognostics. Regarding the importance of data accuracy, a good prognostics requires reliable sources of information. This is why, in this paper, we will first discuss the dependability of wireless sensor networks, and then present a state of the art in prognostic and health management activities.
       


## September
### [Semi-supervised Learning with Deep Generative Models for Asset Failure Prediction](https://arxiv.org/abs/1709.00845)

**Authors:**
Andre S. Yoon, Taehoon Lee, Yongsub Lim, Deokwoo Jung, Philgyun Kang, Dongwon Kim, Keuntae Park, Yongjin Choi

**Abstract:**
This work presents a novel semi-supervised learning approach for data-driven modeling of asset failures when health status is only partially known in historical data. We combine a generative model parameterized by deep neural networks with non-linear embedding technique. It allows us to build prognostic models with the limited amount of health status information for the precise prediction of future asset reliability. The proposed method is evaluated on a publicly available dataset for remaining useful life (RUL) estimation, which shows significant improvement even when a fraction of the data with known health status is as sparse as 1% of the total. Our study suggests that the non-linear embedding based on a deep generative model can efficiently regularize a complex model with deep architectures while achieving high prediction accuracy that is far less sensitive to the availability of health status information.
       


### [Predicting Remaining Useful Life using Time Series Embeddings based on Recurrent Neural Networks](https://arxiv.org/abs/1709.01073)

**Authors:**
Narendhar Gugulothu, Vishnu TV, Pankaj Malhotra, Lovekesh Vig, Puneet Agarwal, Gautam Shroff

**Abstract:**
We consider the problem of estimating the remaining useful life (RUL) of a system or a machine from sensor data. Many approaches for RUL estimation based on sensor data make assumptions about how machines degrade. Additionally, sensor data from machines is noisy and often suffers from missing values in many practical settings. We propose Embed-RUL: a novel approach for RUL estimation from sensor data that does not rely on any degradation-trend assumptions, is robust to noise, and handles missing values. Embed-RUL utilizes a sequence-to-sequence model based on Recurrent Neural Networks (RNNs) to generate embeddings for multivariate time series subsequences. The embeddings for normal and degraded machines tend to be different, and are therefore found to be useful for RUL estimation. We show that the embeddings capture the overall pattern in the time series while filtering out the noise, so that the embeddings of two machines with similar operational behavior are close to each other, even when their sensor readings have significant and varying levels of noise content. We perform experiments on publicly available turbofan engine dataset and a proprietary real-world dataset, and demonstrate that Embed-RUL outperforms the previously reported state-of-the-art on several metrics.
       


### [Real-time predictive maintenance for wind turbines using Big Data frameworks](https://arxiv.org/abs/1709.07250)

**Authors:**
Mikel Canizo, Enrique Onieva, Angel Conde, Santiago Charramendieta, Salvador Trujillo

**Abstract:**
This work presents the evolution of a solution for predictive maintenance to a Big Data environment. The proposed adaptation aims for predicting failures on wind turbines using a data-driven solution deployed in the cloud and which is composed by three main modules. (i) A predictive model generator which generates predictive models for each monitored wind turbine by means of Random Forest algorithm. (ii) A monitoring agent that makes predictions every 10 minutes about failures in wind turbines during the next hour. Finally, (iii) a dashboard where given predictions can be visualized. To implement the solution Apache Spark, Apache Kafka, Apache Mesos and HDFS have been used. Therefore, we have improved the previous work in terms of data process speed, scalability and automation. In addition, we have provided fault-tolerant functionality with a centralized access point from where the status of all the wind turbines of a company localized all over the world can be monitored, reducing O&M costs.
       


## October
### [Driving with Data: Modeling and Forecasting Vehicle Fleet Maintenance in Detroit](https://arxiv.org/abs/1710.06839)

**Authors:**
Josh Gardner, Danai Koutra, Jawad Mroueh, Victor Pang, Arya Farahi, Sam Krassenstein, Jared Webb

**Abstract:**
The City of Detroit maintains an active fleet of over 2500 vehicles, spending an annual average of over \$5 million on new vehicle purchases and over \$7.7 million on maintaining this fleet. Understanding the existence of patterns and trends in this data could be useful to a variety of stakeholders, particularly as Detroit emerges from Chapter 9 bankruptcy, but the patterns in such data are often complex and multivariate and the city lacks dedicated resources for detailed analysis of this data. This work, a data collaboration between the Michigan Data Science Team (http://midas.umich.edu/mdst) and the City of Detroit's Operations and Infrastructure Group, seeks to address this unmet need by analyzing data from the City of Detroit's entire vehicle fleet from 2010-2017. We utilize tensor decomposition techniques to discover and visualize unique temporal patterns in vehicle maintenance; apply differential sequence mining to demonstrate the existence of common and statistically unique maintenance sequences by vehicle make and model; and, after showing these time-dependencies in the dataset, demonstrate an application of a predictive Long Short Term Memory (LSTM) neural network model to predict maintenance sequences. Our analysis shows both the complexities of municipal vehicle fleet data and useful techniques for mining and modeling such data.
       


## November
### [How Long Will My Phone Battery Last?](https://arxiv.org/abs/1711.03651)

**Authors:**
Liang He, Kang G. Shin

**Abstract:**
Mobile devices are only as useful as their battery lasts. Unfortunately, the operation and life of a mobile device's battery degrade over time and usage. The state-of-health (SoH) of batteries quantifies their degradation, but mobile devices are unable to support its accurate estimation -- despite its importance -- due mainly to their limited hardware and dynamic usage patterns, causing various problems such as unexpected device shutoffs or even fire/explosion. To remedy this lack of support, we design, implement and evaluate V-Health, a low-cost user-level SoH estimation service for mobile devices based only on their battery voltage, which is commonly available on all commodity mobile devices. V-Health also enables four novel use-cases that improve mobile users' experience from different perspectives. The design of V-Health is inspired by our empirical finding that the relaxing voltages of a device battery fingerprint its SoH, and is steered by extensive measurements with 15 batteries used for various commodity mobile devices, such as Nexus 6P, Galaxy S3, iPhone 6 Plus, etc. These measurements consist of 13,377 battery discharging/charging/resting cycles and have been conducted over 72 months cumulatively. V-Health has been evaluated via both laboratory experiments and field tests over 4-6 months, showing <5% error in SoH estimation.
       


### [The Health Status of a Population estimated: The History of Health State Curves](https://arxiv.org/abs/1711.09489)

**Authors:**
Christos H Skiadas, Charilaos Skiadas

**Abstract:**
Following the recent publication of our book on Exploring the Health State of a Population by Dynamic Modeling Methods in The Springer Series on Demographic Methods and Population Analysis (DOI 10.1007/978-3-319-65142-2) we provide this brief presentation of the main findings and improvements regarding the Health State of a Population. (See at: http://www.springer.com/gp/book/9783319651415). Here the brief history of the Health State or Health Status curves for individuals and populations is presented including the main references and important figures along with an illustrated Poster (see Figure 13 and http://www.smtda.net/demographics2018.html). Although the Survival Curve is known as long as the life tables have introduced, the Health State Curve was calculated after the introduction of the advanced stochastic theory of the first exit time. The health state curve is illustrated in several graphs either as a fit curve to data or produced after a large number of stochastic realizations. The Health State, the Life Expectancy and the age at mean zero health state are also estimated. Keywords: Health State and Survival Curves, Health status of a population, First exit time stochastic theory, stochastic simulations of health state, Age at Maximum Curvature, Healthy Life Expectancy and HALE, Standard Deviation, Health State Curves, Maximum human lifespan and other.
       


# 2018
## January
### [Efficient Machine-type Communication using Multi-metric Context-awareness for Cars used as Mobile Sensors in Upcoming 5G Networks](https://arxiv.org/abs/1801.03290)

**Authors:**
Benjamin Sliwa, Thomas Liebig, Robert Falkenberg, Johannes Pillmann, Christian Wietfeld

**Abstract:**
Upcoming 5G-based communication networks will be confronted with huge increases in the amount of transmitted sensor data related to massive deployments of static and mobile Internet of Things (IoT) systems. Cars acting as mobile sensors will become important data sources for cloud-based applications like predictive maintenance and dynamic traffic forecast. Due to the limitation of available communication resources, it is expected that the grows in Machine-Type Communication (MTC) will cause severe interference with Human-to-human (H2H) communication. Consequently, more efficient transmission methods are highly required. In this paper, we present a probabilistic scheme for efficient transmission of vehicular sensor data which leverages favorable channel conditions and avoids transmissions when they are expected to be highly resource-consuming. Multiple variants of the proposed scheme are evaluated in comprehensive realworld experiments. Through machine learning based combination of multiple context metrics, the proposed scheme is able to achieve up to 164% higher average data rate values for sensor applications with soft deadline requirements compared to regular periodic transmission.
       


## February
### [Application of Multivariate Data Analysis to machine power measurements as a means of tool life Predictive Maintenance for reducing product waste](https://arxiv.org/abs/1802.08338)

**Authors:**
Darren A Whitaker, David Egan, Eoin OBrien, David Kinnear

**Abstract:**
Modern manufacturing industries are increasingly looking to predictive analytics to gain decision making information from process data. This is driven by high levels of competition and a need to reduce operating costs. The presented work takes data in the form of a power measurement recorded during a medical device manufacturing process and uses multivariate data analysis (MVDA) to extract information leading to the proposal of a predictive maintenance scheduling algorithm. The proposed MVDA model was able to predict with 100 % accuracy the condition of a grinding tool.
       


### [Parsimonious Network based on Fuzzy Inference System (PANFIS) for Time Series Feature Prediction of Low Speed Slew Bearing Prognosis](https://arxiv.org/abs/1802.09332)

**Authors:**
Wahyu Caesarendra, Mahardhika Pratama, Tegoeh Tjahjowidodo, Kiet Tieud, Buyung Kosasih

**Abstract:**
In recent years, the utilization of rotating parts, e.g. bearings and gears, has been continuously supporting the manufacturing line to produce consistent output quality. Due to their critical role, the breakdown of these components might significantly impact the production rate. A proper condition based monitoring (CBM) is among a few ways to maintain and monitor the rotating systems. Prognosis, as one of the major tasks in CBM that predicts and estimates the remaining useful life of the machine, has attracted significant interest in decades. This paper presents a literature review on prognosis approaches from published papers in the last decade. The prognostic approaches are described comprehensively to provide a better idea on how to select an appropriate prognosis method for specific needs. An advanced predictive analytics, namely Parsimonious Network Based on Fuzzy Inference System (PANFIS), was proposed and tested into the low speed slew bearing data. PANFIS differs itself from conventional prognostic approaches in which it supports for online lifelong prognostics without the requirement of retraining or reconfiguration phase. The method is applied to normal-to-failure bearing vibration data collected for 139 days and to predict the time-domain features of vibration slew bearing signals. The performance of the proposed method is compared to some established methods such as ANFIS, eTS, and Simp_eTS. From the results, it is suggested that PANFIS offers outstanding performance compared to those of other methods.
       


## April
### [Anomaly Detection for Industrial Big Data](https://arxiv.org/abs/1804.02998)

**Authors:**
Neil Caithness, David Wallom

**Abstract:**
As the Industrial Internet of Things (IIoT) grows, systems are increasingly being monitored by arrays of sensors returning time-series data at ever-increasing 'volume, velocity and variety' (i.e. Industrial Big Data). An obvious use for these data is real-time systems condition monitoring and prognostic time to failure analysis (remaining useful life, RUL). (e.g. See white papers by Senseye.io, and output of the NASA Prognostics Center of Excellence (PCoE).) However, as noted by Agrawal and Choudhary 'Our ability to collect "big data" has greatly surpassed our capability to analyze it, underscoring the emergence of the fourth paradigm of science, which is data-driven discovery.' In order to fully utilize the potential of Industrial Big Data we need data-driven techniques that operate at scales that process models cannot. Here we present a prototype technique for data-driven anomaly detection to operate at industrial scale. The method generalizes to application with almost any multivariate dataset based on independent ordinations of repeated (bootstrapped) partitions of the dataset and inspection of the joint distribution of ordinal distances.
       


### [Deep Learning on Key Performance Indicators for Predictive Maintenance in SAP HANA](https://arxiv.org/abs/1804.05497)

**Authors:**
Jaekoo Lee, Byunghan Lee, Jongyoon Song, Jaesik Yoon, Yongsik Lee, Donghun Lee, Sungroh Yoon

**Abstract:**
With a new era of cloud and big data, Database Management Systems (DBMSs) have become more crucial in numerous enterprise business applications in all the industries. Accordingly, the importance of their proactive and preventive maintenance has also increased. However, detecting problems by predefined rules or stochastic modeling has limitations, particularly when analyzing the data on high-dimensional Key Performance Indicators (KPIs) from a DBMS. In recent years, Deep Learning (DL) has opened new opportunities for this complex analysis. In this paper, we present two complementary DL approaches to detect anomalies in SAP HANA. A temporal learning approach is used to detect abnormal patterns based on unlabeled historical data, whereas a spatial learning approach is used to classify known anomalies based on labeled data. We implement a system in SAP HANA integrated with Google TensorFlow. The experimental results with real-world data confirm the effectiveness of the system and models.
       


### [Building robust prediction models for defective sensor data using Artificial Neural Networks](https://arxiv.org/abs/1804.05544)

**Authors:**
Arvind Kumar Shekar, Cláudio Rebelo de Sá, Hugo Ferreira, Carlos Soares

**Abstract:**
Predicting the health of components in complex dynamic systems such as an automobile poses numerous challenges. The primary aim of such predictive systems is to use the high-dimensional data acquired from different sensors and predict the state-of-health of a particular component, e.g., brake pad. The classical approach involves selecting a smaller set of relevant sensor signals using feature selection and using them to train a machine learning algorithm. However, this fails to address two prominent problems: (1) sensors are susceptible to failure when exposed to extreme conditions over a long periods of time; (2) sensors are electrical devices that can be affected by noise or electrical interference. Using the failed and noisy sensor signals as inputs largely reduce the prediction accuracy. To tackle this problem, it is advantageous to use the information from all sensor signals, so that the failure of one sensor can be compensated by another. In this work, we propose an Artificial Neural Network (ANN) based framework to exploit the information from a large number of signals. Secondly, our framework introduces a data augmentation approach to perform accurate predictions in spite of noisy signals. The plausibility of our framework is validated on real life industrial application from Robert Bosch GmbH.
       


### [A Parallel/Distributed Algorithmic Framework for Mining All Quantitative Association Rules](https://arxiv.org/abs/1804.06764)

**Authors:**
Ioannis T. Christou, Emmanouil Amolochitis, Zheng-Hua Tan

**Abstract:**
We present QARMA, an efficient novel parallel algorithm for mining all Quantitative Association Rules in large multidimensional datasets where items are required to have at least a single common attribute to be specified in the rules single consequent item. Given a minimum support level and a set of threshold criteria of interestingness measures such as confidence, conviction etc. our algorithm guarantees the generation of all non-dominated Quantitative Association Rules that meet the minimum support and interestingness requirements. Such rules can be of great importance to marketing departments seeking to optimize targeted campaigns, or general market segmentation. They can also be of value in medical applications, financial as well as predictive maintenance domains. We provide computational results showing the scalability of our algorithm, and its capability to produce all rules to be found in large scale synthetic and real world datasets such as Movie Lens, within a few seconds or minutes of computational time on commodity hardware.
       


## May
### [An Online RFID Localization in the Manufacturing Shopfloor](https://arxiv.org/abs/1805.07715)

**Authors:**
Andri Ashfahani, Mahardhika Pratama, Edwin Lughofer, Qing Cai, Huang Sheng

**Abstract:**
{Radio Frequency Identification technology has gained popularity for cheap and easy deployment. In the realm of manufacturing shopfloor, it can be used to track the location of manufacturing objects to achieve better efficiency. The underlying challenge of localization lies in the non-stationary characteristics of manufacturing shopfloor which calls for an adaptive life-long learning strategy in order to arrive at accurate localization results. This paper presents an evolving model based on a novel evolving intelligent system, namely evolving Type-2 Quantum Fuzzy Neural Network (eT2QFNN), which features an interval type-2 quantum fuzzy set with uncertain jump positions. The quantum fuzzy set possesses a graded membership degree which enables better identification of overlaps between classes. The eT2QFNN works fully in the evolving mode where all parameters including the number of rules are automatically adjusted and generated on the fly. The parameter adjustment scenario relies on decoupled extended Kalman filter method. Our numerical study shows that eT2QFNN is able to deliver comparable accuracy compared to state-of-the-art algorithms.
       


## June
### [Predictive Maintenance for Industrial IoT of Vehicle Fleets using Hierarchical Modified Fuzzy Support Vector Machine](https://arxiv.org/abs/1806.09612)

**Author:**
Arindam Chaudhuri

**Abstract:**
Connected vehicle fleets are deployed worldwide in several industrial IoT scenarios. With the gradual increase of machines being controlled and managed through networked smart devices, the predictive maintenance potential grows rapidly. Predictive maintenance has the potential of optimizing uptime as well as performance such that time and labor associated with inspections and preventive maintenance are reduced. In order to understand the trends of vehicle faults with respect to important vehicle attributes viz mileage, age, vehicle type etc this problem is addressed through hierarchical modified fuzzy support vector machine (HMFSVM). The proposed method is compared with other commonly used approaches like logistic regression, random forests and support vector machines. This helps better implementation of telematics data to ensure preventative management as part of the desired solution. The superiority of the proposed method is highlighted through several experimental results.
       


## July
### [Prognostics Estimations with Dynamic States](https://arxiv.org/abs/1807.06093)

**Authors:**
Rong-Jing Bao, Hai-Jun Rong, Zhi-Xin Yang, Badong Chen

**Abstract:**
The health state assessment and remaining useful life (RUL) estimation play very important roles in prognostics and health management (PHM), owing to their abilities to reduce the maintenance and improve the safety of machines or equipment. However, they generally suffer from this problem of lacking prior knowledge to pre-define the exact failure thresholds for a machinery operating in a dynamic environment with a high level of uncertainty. In this case, dynamic thresholds depicted by the discrete states is a very attractive way to estimate the RUL of a dynamic machinery. Currently, there are only very few works considering the dynamic thresholds, and these studies adopted different algorithms to determine the discrete states and predict the continuous states separately, which largely increases the complexity of the learning process. In this paper, we propose a novel prognostics approach for RUL estimation of aero-engines with self-joint prediction of continuous and discrete states, wherein the prediction of continuous and discrete states are conducted simultaneously and dynamically within one learning framework.
       


## August
### [Leveraging Knowledge Graph Embedding Techniques for Industry 4.0 Use Cases](https://arxiv.org/abs/1808.00434)

**Authors:**
Martina Garofalo, Maria Angela Pellegrino, Abdulrahman Altabba, Michael Cochez

**Abstract:**
Industry is evolving towards Industry 4.0, which holds the promise of increased flexibility in manufacturing, better quality and improved productivity. A core actor of this growth is using sensors, which must capture data that can used in unforeseen ways to achieve a performance not achievable without them. However, the complexity of this improved setting is much greater than what is currently used in practice. Hence, it is imperative that the management cannot only be performed by human labor force, but part of that will be done by automated algorithms instead. A natural way to represent the data generated by this large amount of sensors, which are not acting measuring independent variables, and the interaction of the different devices is by using a graph data model. Then, machine learning could be used to aid the Industry 4.0 system to, for example, perform predictive maintenance. However, machine learning directly on graphs, needs feature engineering and has scalability issues. In this paper we discuss methods to convert (embed) the graph in a vector space, such that it becomes feasible to use traditional machine learning methods for Industry 4.0 settings.
       


### [Steady State Reduction of generalized Lotka-Volterra systems in the microbiome](https://arxiv.org/abs/1808.01715)

**Authors:**
Eric W. Jones, Jean M. Carlson

**Abstract:**
The generalized Lotka-Volterra (gLV) equations, a classic model from theoretical ecology, describe the population dynamics of a set of interacting species. As the number of species in these systems grow in number, their dynamics become increasingly complex and intractable. We introduce Steady State Reduction (SSR), a method that reduces a gLV system of many ecological species into two-dimensional (2D) subsystems that each obey gLV dynamics and whose basis vectors are steady states of the high-dimensional model. We apply this method to an experimentally-derived model of the gut microbiome in order to observe the transition between "healthy" and "diseased" microbial states. Specifically, we use SSR to investigate how fecal microbiota transplantation, a promising clinical treatment for dysbiosis, can revert a diseased microbial state to health.
       


## September
### [Domain Adaptation for Robot Predictive Maintenance Systems](https://arxiv.org/abs/1809.08626)

**Authors:**
Arash Golibagh Mahyari, Thomas Locker

**Abstract:**
Industrial robots play an increasingly important role in a growing number of fields. For example, robotics is used to increase productivity while reducing costs in various aspects of manufacturing. Since robots are often set up in production lines, the breakdown of a single robot has a negative impact on the entire process, in the worst case bringing the whole line to a halt until the issue is resolved, leading to substantial financial losses due to the unforeseen downtime. Therefore, predictive maintenance systems based on the internal signals of robots have gained attention as an essential component of robotics service offerings. The main shortcoming of existing predictive maintenance algorithms is that the extracted features typically differ significantly from the learnt model when the operation of the robot changes, incurring false alarms. In order to mitigate this problem, predictive maintenance algorithms require the model to be retrained with normal data of the new operation. In this paper, we propose a novel solution based on transfer learning to pass the knowledge of the trained model from one operation to another in order to prevent the need for retraining and to eliminate such false alarms. The deployment of the proposed unsupervised transfer learning algorithm on real-world datasets demonstrates that the algorithm can not only distinguish between operation and mechanical condition change, it further yields a sharper deviation from the trained model in case of a mechanical condition change and thus detects mechanical issues with higher confidence.
       


### [Cost-Sensitive Learning for Predictive Maintenance](https://arxiv.org/abs/1809.10979)

**Authors:**
Stephan Spiegel, Fabian Mueller, Dorothea Weismann, John Bird

**Abstract:**
In predictive maintenance, model performance is usually assessed by means of precision, recall, and F1-score. However, employing the model with best performance, e.g. highest F1-score, does not necessarily result in minimum maintenance cost, but can instead lead to additional expenses. Thus, we propose to perform model selection based on the economic costs associated with the particular maintenance application. We show that cost-sensitive learning for predictive maintenance can result in significant cost reduction and fault tolerant policies, since it allows to incorporate various business constraints and requirements.
       


## October
### [Temporal Convolutional Memory Networks for Remaining Useful Life Estimation of Industrial Machinery](https://arxiv.org/abs/1810.05644)

**Authors:**
Lahiru Jayasinghe, Tharaka Samarasinghe, Chau Yuen, Jenny Chen Ni Low, Shuzhi Sam Ge

**Abstract:**
Accurately estimating the remaining useful life (RUL) of industrial machinery is beneficial in many real-world applications. Estimation techniques have mainly utilized linear models or neural network based approaches with a focus on short term time dependencies. This paper, introduces a system model that incorporates temporal convolutions with both long term and short term time dependencies. The proposed network learns salient features and complex temporal variations in sensor values, and predicts the RUL. A data augmentation method is used for increased accuracy. The proposed method is compared with several state-of-the-art algorithms on publicly available datasets. It demonstrates promising results, with superior results for datasets obtained from complex environments.
       


### [Joint Optimization of Opportunistic Predictive Maintenance and Multi-location Spare Part Inventories for a Deteriorating System Considering Imperfect Actions](https://arxiv.org/abs/1810.06315)

**Author:**
Morteza Soltani

**Abstract:**
Considering the close interaction between spare parts logistics and maintenance planning, this paper presents a model for joint optimization of multi-location spare parts supply chain and condition-based maintenance under predictive and opportunistic approaches. Simultaneous use of the imperfect maintenance actions and innovative policy on spare part ordering, which is defined based on the deterioration characteristic of the system, is a significant contribution to the research. This paper also proposes the method to determine the inspection time which not only considers restraints of the both maintenance and spare parts provision policies, but also uses an event-driven approach in order to prevent unnecessary inspections. Defined decision variables such reliability, upper limit for spare parts order quantity, preventive maintenance threshold, re-ordering level of degradation, and the maximum level of successive imperfect actions will be optimized via stochastic Monte-Carlo simulation. The optimization follows two objectives: (1) system should reach the expected availability which helps decision makers apply the opportunistic approach (2) and cost rate function as an objective function must be minimized. To illustrate the use of the proposed model, a numerical example and its results finally is presented. Key words: maintenance, multi-location supply chain, spare parts inventory, imperfect maintenance, predictive inspection, opportunistic approach, availability, reliability
       


### [Mechanisms for Integrated Feature Normalization and Remaining Useful Life Estimation Using LSTMs Applied to Hard-Disks](https://arxiv.org/abs/1810.08985)

**Authors:**
Sanchita Basak, Saptarshi Sengupta, Abhishek Dubey

**Abstract:**
With emerging smart communities, improving overall system availability is becoming a major concern. In order to improve the reliability of the components in a system we propose an inference model to predict Remaining Useful Life (RUL) of those components. In this paper we work with components of backend data servers such as hard disks, that are subject to degradation. A Deep Long-Short Term Memory (LSTM) Network is used as the backbone of this fast, data-driven decision framework and dynamically captures the pattern of the incoming data. In the article, we discuss the architecture of the neural network and describe the mechanisms to choose the various hyper-parameters. Further, we describe the challenges faced in extracting effective training sets from highly unorganized and class-imbalanced big data and establish methods for online predictions with extensive data pre-processing, feature extraction and validation through online simulation sets with unknown remaining useful lives of the hard disks. Our algorithm performs especially well in predicting RUL near the critical zone of a device approaching failure. With the proposed approach we are able to predict whether a disk is going to fail in next ten days with an average precision of 0.8435. We also show that the architecture trained on a particular model can be used to predict RUL for devices in different models from same manufacturer through transfer learning.
       


### [Post-prognostics decision in Cyber-Physical Systems](https://arxiv.org/abs/1810.11732)

**Authors:**
Safa Meraghni, Labib Sadek Terrissa, Soheyb Ayad, Noureddine Zerhouni, Christophe Varnier

**Abstract:**
Prognostics and Health Management (PHM) offers several benefits for predictive maintenance. It predicts the future behavior of a system as well as its Remaining Useful Life (RUL). This RUL is used to planned the maintenance operation to avoid the failure, the stop time and optimize the cost of the maintenance and failure. However, with the development of the industry the assets are nowadays distributed this is why the PHM needs to be developed using the new IT. In our work we propose a PHM solution based on Cyber physical system where the physical side is connected to the analyze process of the PHM which are developed in the cloud to be shared and to benefit of the cloud characteristics
       


## November
### [ADEPOS: Anomaly Detection based Power Saving for Predictive Maintenance using Edge Computing](https://arxiv.org/abs/1811.00873)

**Authors:**
Sumon Kumar Bose, Bapi Kar, Mohendra Roy, Pradeep Kumar Gopalakrishnan, Arindam Basu

**Abstract:**
In industry 4.0, predictive maintenance(PM) is one of the most important applications pertaining to the Internet of Things(IoT). Machine learning is used to predict the possible failure of a machine before the actual event occurs. However, the main challenges in PM are (a) lack of enough data from failing machines, and (b) paucity of power and bandwidth to transmit sensor data to cloud throughout the lifetime of the machine. Alternatively, edge computing approaches reduce data transmission and consume low energy. In this paper, we propose Anomaly Detection based Power Saving(ADEPOS) scheme using approximate computing through the lifetime of the machine. In the beginning of the machines life, low accuracy computations are used when the machine is healthy. However, on the detection of anomalies, as time progresses, the system is switched to higher accuracy modes. We show using the NASA bearing dataset that using ADEPOS, we need 8.8X less neurons on average and based on post-layout results, the resultant energy savings are 6.4 to 6.65X
       


### [Linear Programming for Decision Processes with Partial Information](https://arxiv.org/abs/1811.08880)

**Authors:**
Victor Cohen, Axel Parmentier

**Abstract:**
Markov Decision Processes (MDPs) are stochastic optimization problems that model situations where a decision maker controls a system based on its state. Partially observed Markov decision processes (POMDPs) are generalizations of MDPs where the decision maker has only partial information on the state of the system. Decomposable POMDPs are specific cases of POMDPs that enable one to model systems with several components. Such problems naturally model a wide range of applications such as predictive maintenance. Finding an optimal policy for a POMDP is PSPACE-hard and practically challenging. We introduce a mixed integer linear programming (MILP) formulation for POMDPs restricted to the policies that only depend on the current observation, as well as valid inequalities that are based on a probabilistic interpretation of the dependence between variables. The linear relaxation provides a good bound for the usual POMDPs where the policies depend on the full history of observations and actions. Solving decomposable POMDPs is especially challenging due to the curse of dimensionality. Leveraging our MILP formulation for POMDPs, we introduce a linear program based on ``fluid formulation'' for decomposable POMDPs, that provides both a bound on the optimal value and a practically efficient heuristic to find a good policy. Numerical experiments show the efficiency of our approaches to POMDPs and decomposable POMDPs.
       


## December
### [A deep learning-based remaining useful life prediction approach for bearings](https://arxiv.org/abs/1812.03315)

**Authors:**
Cheng Cheng, Guijun Ma, Yong Zhang, Mingyang Sun, Fei Teng, Han Ding, Ye Yuan

**Abstract:**
In industrial applications, nearly half the failures of motors are caused by the degradation of rolling element bearings (REBs). Therefore, accurately estimating the remaining useful life (RUL) for REBs are of crucial importance to ensure the reliability and safety of mechanical systems. To tackle this challenge, model-based approaches are often limited by the complexity of mathematical modeling. Conventional data-driven approaches, on the other hand, require massive efforts to extract the degradation features and construct health index. In this paper, a novel online data-driven framework is proposed to exploit the adoption of deep convolutional neural networks (CNN) in predicting the RUL of bearings. More concretely, the raw vibrations of training bearings are first processed using the Hilbert-Huang transform (HHT) and a novel nonlinear degradation indicator is constructed as the label for learning. The CNN is then employed to identify the hidden pattern between the extracted degradation indicator and the vibration of training bearings, which makes it possible to estimate the degradation of the test bearings automatically. Finally, testing bearings' RULs are predicted by using a $ε$-support vector regression model. The superior performance of the proposed RUL estimation framework, compared with the state-of-the-art approaches, is demonstrated through the experimental results. The generality of the proposed CNN model is also validated by transferring to bearings undergoing different operating conditions.
       


### [Data Strategies for Fleetwide Predictive Maintenance](https://arxiv.org/abs/1812.04446)

**Author:**
David Noever

**Abstract:**
For predictive maintenance, we examine one of the largest public datasets for machine failures derived along with their corresponding precursors as error rates, historical part replacements, and sensor inputs. To simplify the time and accuracy comparison between 27 different algorithms, we treat the imbalance between normal and failing states with nominal under-sampling. We identify 3 promising regression and discriminant algorithms with both higher accuracy (96%) and twenty-fold faster execution times than previous work. Because predictive maintenance success hinges on input features prior to prediction, we provide a methodology to rank-order feature importance and show that for this dataset, error counts prove more predictive than scheduled maintenance might imply solely based on more traditional factors such as machine age or last replacement times.
       


### [seq2graph: Discovering Dynamic Dependencies from Multivariate Time Series with Multi-level Attention](https://arxiv.org/abs/1812.04448)

**Authors:**
Xuan-Hong Dang, Syed Yousaf Shah, Petros Zerfos

**Abstract:**
Discovering temporal lagged and inter-dependencies in multivariate time series data is an important task. However, in many real-world applications, such as commercial cloud management, manufacturing predictive maintenance, and portfolios performance analysis, such dependencies can be non-linear and time-variant, which makes it more challenging to extract such dependencies through traditional methods such as Granger causality or clustering. In this work, we present a novel deep learning model that uses multiple layers of customized gated recurrent units (GRUs) for discovering both time lagged behaviors as well as inter-timeseries dependencies in the form of directed weighted graphs. We introduce a key component of Dual-purpose recurrent neural network that decodes information in the temporal domain to discover lagged dependencies within each time series, and encodes them into a set of vectors which, collected from all component time series, form the informative inputs to discover inter-dependencies. Though the discovery of two types of dependencies are separated at different hierarchical levels, they are tightly connected and jointly trained in an end-to-end manner. With this joint training, learning of one type of dependency immediately impacts the learning of the other one, leading to overall accurate dependencies discovery. We empirically test our model on synthetic time series data in which the exact form of (non-linear) dependencies is known. We also evaluate its performance on two real-world applications, (i) performance monitoring data from a commercial cloud provider, which exhibit highly dynamic, non-linear, and volatile behavior and, (ii) sensor data from a manufacturing plant. We further show how our approach is able to capture these dependency behaviors via intuitive and interpretable dependency graphs and use them to generate highly accurate forecasts.
       


### [Two Birds with One Network: Unifying Failure Event Prediction and Time-to-failure Modeling](https://arxiv.org/abs/1812.07142)

**Authors:**
Karan Aggarwal, Onur Atan, Ahmed Farahat, Chi Zhang, Kosta Ristovski, Chetan Gupta

**Abstract:**
One of the key challenges in predictive maintenance is to predict the impending downtime of an equipment with a reasonable prediction horizon so that countermeasures can be put in place. Classically, this problem has been posed in two different ways which are typically solved independently: (1) Remaining useful life (RUL) estimation as a long-term prediction task to estimate how much time is left in the useful life of the equipment and (2) Failure prediction (FP) as a short-term prediction task to assess the probability of a failure within a pre-specified time window. As these two tasks are related, performing them separately is sub-optimal and might results in inconsistent predictions for the same equipment. In order to alleviate these issues, we propose two methods: Deep Weibull model (DW-RNN) and multi-task learning (MTL-RNN). DW-RNN is able to learn the underlying failure dynamics by fitting Weibull distribution parameters using a deep neural network, learned with a survival likelihood, without training directly on each task. While DW-RNN makes an explicit assumption on the data distribution, MTL-RNN exploits the implicit relationship between the long-term RUL and short-term FP tasks to learn the underlying distribution. Additionally, both our methods can leverage the non-failed equipment data for RUL estimation. We demonstrate that our methods consistently outperform baseline RUL methods that can be used for FP while producing consistent results for RUL and FP. We also show that our methods perform at par with baselines trained on the objectives optimized for either of the two tasks.
       


# 2019
## January
### [Individual common dolphin identification via metric embedding learning](https://arxiv.org/abs/1901.03662)

**Authors:**
Soren Bouma, Matthew D. M. Pawley, Krista Hupman, Andrew Gilman

**Abstract:**
Photo-identification (photo-id) of dolphin individuals is a commonly used technique in ecological sciences to monitor state and health of individuals, as well as to study the social structure and distribution of a population. Traditional photo-id involves a laborious manual process of matching each dolphin fin photograph captured in the field to a catalogue of known individuals.
  We examine this problem in the context of open-set recognition and utilise a triplet loss function to learn a compact representation of fin images in a Euclidean embedding, where the Euclidean distance metric represents fin similarity. We show that this compact representation can be successfully learnt from a fairly small (in deep learning context) training set and still generalise well to out-of-sample identities (completely new dolphin individuals), with top-1 and top-5 test set (37 individuals) accuracy of $90.5\pm2$ and $93.6\pm1$ percent. In the presence of 1200 distractors, top-1 accuracy dropped by $12\%$; however, top-5 accuracy saw only a $2.8\%$ drop
       


### [Fleet Prognosis with Physics-informed Recurrent Neural Networks](https://arxiv.org/abs/1901.05512)

**Authors:**
Renato Giorgiani Nascimento, Felipe A. C. Viana

**Abstract:**
Services and warranties of large fleets of engineering assets is a very profitable business. The success of companies in that area is often related to predictive maintenance driven by advanced analytics. Therefore, accurate modeling, as a way to understand how the complex interactions between operating conditions and component capability define useful life, is key for services profitability. Unfortunately, building prognosis models for large fleets is a daunting task as factors such as duty cycle variation, harsh environments, inadequate maintenance, and problems with mass production can lead to large discrepancies between designed and observed useful lives. This paper introduces a novel physics-informed neural network approach to prognosis by extending recurrent neural networks to cumulative damage models. We propose a new recurrent neural network cell designed to merge physics-informed and data-driven layers. With that, engineers and scientists have the chance to use physics-informed layers to model parts that are well understood (e.g., fatigue crack growth) and use data-driven layers to model parts that are poorly characterized (e.g., internal loads). A simple numerical experiment is used to present the main features of the proposed physics-informed recurrent neural network for damage accumulation. The test problem consist of predicting fatigue crack length for a synthetic fleet of airplanes subject to different mission mixes. The model is trained using full observation inputs (far-field loads) and very limited observation of outputs (crack length at inspection for only a portion of the fleet). The results demonstrate that our proposed hybrid physics-informed recurrent neural network is able to accurately model fatigue crack growth even when the observed distribution of crack length does not match with the (unobservable) fleet distribution.
       


### [The Sequential Algorithm for Combined State of Charge and State of Health Estimation of Lithium Ion Battery based on Active Current Injection](https://arxiv.org/abs/1901.06000)

**Authors:**
Ziyou Song, Jun Hou, Xuefeng Li, Xiaogang Wu, Xiaosong Hu, Heath Hofmann, Jing Sun

**Abstract:**
When State of Charge, State of Health, and parameters of the Lithium-ion battery are estimated simultaneously, the estimation accuracy is hard to be ensured due to uncertainties in the estimation process. To improve the estimation performance a sequential algorithm, which uses frequency scale separation and estimates parameters/states sequentially by injecting currents with different frequencies, is proposed in this paper. Specifically, by incorporating a high-pass filter, the parameters can be independently characterized by injecting high-frequency and medium-frequency currents, respectively. Using the estimated parameters, battery capacity and State of Charge can then be estimated concurrently. Experimental results show that the estimation accuracy of the proposed sequential algorithm is much better than the concurrent algorithm where all parameters/states are estimated simultaneously, and the computational cost can also be reduced. Finally, experiments are conducted under different temperatures to verify the effectiveness of the proposed algorithm for various battery capacities.
       


### [Are Smart Contracts and Blockchains Suitable for Decentralized Railway Control?](https://arxiv.org/abs/1901.06236)

**Authors:**
Michael Kuperberg, Daniel Kindler, Sabina Jeschke

**Abstract:**
Conventional railway operations employ specialized software and hardware to ensure safe and secure train operations. Track occupation and signaling are governed by central control offices, while trains (and their drivers) receive instructions. To make this setup more dynamic, the train operations can be decentralized by enabling the trains to find routes and make decisions which are safeguarded and protocolled in an auditable manner. In this paper, we present the findings of a first-of-its-kind blockchain-based prototype implementation for railway control, based on decentralization but also ensuring that the overall system state remains conflict-free and safe. We also show how a blockchain-based approach simplifies usage billing and enables a train-to-train/machine-to-machine economy. Finally, first ideas addressing the use of blockchain as a life-cycle approach for condition based monitoring and predictive maintenance in train operations are outlined.
       


### [Predictive Maintenance in Photovoltaic Plants with a Big Data Approach](https://arxiv.org/abs/1901.10855)

**Authors:**
Alessandro Betti, Maria Luisa Lo Trovato, Fabio Salvatore Leonardi, Giuseppe Leotta, Fabrizio Ruffini, Ciro Lanzetta

**Abstract:**
This paper presents a novel and flexible solution for fault prediction based on data collected from SCADA system. Fault prediction is offered at two different levels based on a data-driven approach: (a) generic fault/status prediction and (b) specific fault class prediction, implemented by means of two different machine learning based modules built on an unsupervised clustering algorithm and a Pattern Recognition Neural Network, respectively. Model has been assessed on a park of six photovoltaic (PV) plants up to 10 MW and on more than one hundred inverter modules of three different technology brands. The results indicate that the proposed method is effective in (a) predicting incipient generic faults up to 7 days in advance with sensitivity up to 95% and (b) anticipating damage of specific fault classes with times ranging from few hours up to 7 days. The model is easily deployable for on-line monitoring of anomalies on new PV plants and technologies, requiring only the availability of historical SCADA and fault data, fault taxonomy and inverter electrical datasheet. Keywords: Data Mining, Fault Prediction, Inverter Module, Key Performance Indicator, Lost Production
       


## February
### [Automatic Hyperparameter Tuning Method for Local Outlier Factor, with Applications to Anomaly Detection](https://arxiv.org/abs/1902.00567)

**Authors:**
Zekun Xu, Deovrat Kakde, Arin Chaudhuri

**Abstract:**
In recent years, there have been many practical applications of anomaly detection such as in predictive maintenance, detection of credit fraud, network intrusion, and system failure. The goal of anomaly detection is to identify in the test data anomalous behaviors that are either rare or unseen in the training data. This is a common goal in predictive maintenance, which aims to forecast the imminent faults of an appliance given abundant samples of normal behaviors. Local outlier factor (LOF) is one of the state-of-the-art models used for anomaly detection, but the predictive performance of LOF depends greatly on the selection of hyperparameters. In this paper, we propose a novel, heuristic methodology to tune the hyperparameters in LOF. A tuned LOF model that uses the proposed method shows good predictive performance in both simulations and real data sets.
       


### [Dictionary learning approach to monitoring of wind turbine drivetrain bearings](https://arxiv.org/abs/1902.01426)

**Authors:**
Sergio Martin-del-Campo, Fredrik Sandin, Daniel Strömbergsson

**Abstract:**
Condition monitoring is central to the efficient operation of wind farms due to the challenging operating conditions, rapid technology development and large number of aging wind turbines. In particular, predictive maintenance planning requires the early detection of faults with few false positives. Achieving this type of detection is a challenging problem due to the complex and weak signatures of some faults, particularly the faults that occur in some of the drivetrain bearings. Here, we investigate recently proposed condition monitoring methods based on unsupervised dictionary learning using vibration data recorded over 46 months under typical industrial operations. Thus, we contribute novel test results and real world data that are made publicly available. The results of former studies addressing condition monitoring tasks using dictionary learning indicate that unsupervised feature learning is useful for diagnosis and anomaly detection purposes. However, these studies are based on small sets of labeled data from test rigs operating under controlled conditions that focus on classification tasks, which are useful for quantitative method comparisons but gives little insight into how useful these approaches are in practice. In this study, dictionaries are learned from gearbox vibrations in six different turbines, and the dictionaries are subsequently propagated over a few years of monitoring data when faults are known to occur. We perform the experiment using two different sparse coding algorithms to investigate if the algorithm selected affects the features of abnormal conditions. We calculate the dictionary distance between the initial and propagated dictionaries and find the time periods of abnormal dictionary adaptation starting six months before a drivetrain bearing replacement and one year before the resulting gearbox replacement.
       


### [Evaluating reliability of complex systems for Predictive maintenance](https://arxiv.org/abs/1902.03495)

**Authors:**
Dongjin Lee, Rong Pan

**Abstract:**
Predictive Maintenance (PdM) can only be implemented when the online knowledge of system condition is available, and this has become available with deployment of on-equipment sensors. To date, most studies on predicting the remaining useful lifetime of a system have been focusing on either single-component systems or systems with deterministic reliability structures. This assumption is not applicable on some realistic problems, where there exist uncertainties in reliability structures of complex systems. In this paper, a PdM scheme is developed by employing a Discrete Time Markov Chain (DTMC) for forecasting the health of monitored components and a Bayesian Network (BN) for modeling the multi-component system reliability. Therefore, probabilistic inferences on both the system and its components status can be made and PdM can be scheduled on both levels.
       


### [KINN: Incorporating Expert Knowledge in Neural Networks](https://arxiv.org/abs/1902.05653)

**Authors:**
Muhammad Ali Chattha, Shoaib Ahmed Siddiqui, Muhammad Imran Malik, Ludger van Elst, Andreas Dengel, Sheraz Ahmed

**Abstract:**
The promise of ANNs to automatically discover and extract useful features/patterns from data without dwelling on domain expertise although seems highly promising but comes at the cost of high reliance on large amount of accurately labeled data, which is often hard to acquire and formulate especially in time-series domains like anomaly detection, natural disaster management, predictive maintenance and healthcare. As these networks completely rely on data and ignore a very important modality i.e. expert, they are unable to harvest any benefit from the expert knowledge, which in many cases is very useful. In this paper, we try to bridge the gap between these data driven and expert knowledge based systems by introducing a novel framework for incorporating expert knowledge into the network (KINN). Integrating expert knowledge into the network has three key advantages: (a) Reduction in the amount of data needed to train the model, (b) provision of a lower bound on the performance of the resulting classifier by obtaining the best of both worlds, and (c) improved convergence of model parameters (model converges in smaller number of epochs). Although experts are extremely good in solving different tasks, there are some trends and patterns, which are usually hidden only in the data. Therefore, KINN employs a novel residual knowledge incorporation scheme, which can automatically determine the quality of the predictions made by the expert and rectify it accordingly by learning the trends/patterns from data. Specifically, the method tries to use information contained in one modality to complement information missed by the other. We evaluated KINN on a real world traffic flow prediction problem. KINN significantly superseded performance of both the expert and as well as the base network (LSTM in this case) when evaluated in isolation, highlighting its superiority for the task.
       


### [Seismic Damage Assessment of Instrumented Wood-frame Buildings: A Case-study of NEESWood Full-scale Shake Table Tests](https://arxiv.org/abs/1902.09955)

**Authors:**
Milad Roohi, Eric M. Hernandez, David Rosowsky

**Abstract:**
The authors propose a methodology to perform seismic damage assessment of instrumented wood-frame buildings using response measurements. The proposed methodology employs a nonlinear model-based state observer that combines sparse acceleration measurements and a nonlinear structural model of a building to estimate the complete seismic response including displacements, velocity, acceleration and internal forces in all structural members. From the estimated seismic response and structural characteristics of each shear wall of the building, element-by-element seismic damage indices are computed and remaining useful life (pertaining to seismic effects) is predicted. The methodology is illustrated using measured data from the 2009 NEESWood Capstone full-scale shake table tests at the E-Defense facility in Japan.
       


### [Detection of Gait Asymmetry Using Indoor Doppler Radar](https://arxiv.org/abs/1902.09977)

**Authors:**
Ann-Kathrin Seifert, Abdelhak M. Zoubir, Moeness G. Amin

**Abstract:**
Doppler radar systems enable unobtrusive and privacy-preserving long-term monitoring of human motions indoors. In particular, a person's gait can provide important information about their state of health. Utilizing micro-Doppler signatures, we show that radar is capable of detecting small differences between the step motions of the two legs, which results in asymmetric gait. Image-based and physical features are extracted from the radar return signals of several individuals, including four persons with different diagnosed gait disorders. It is shown that gait asymmetry is correctly detected with high probability, irrespective of the underlying pathology, for at least one motion direction.
       


## March
### [State of health estimation for lithium-ion battery by combining incremental capacity analysis with Gaussian process regression](https://arxiv.org/abs/1903.07672)

**Authors:**
Xiaoyu Li, Zhenpo Wang

**Abstract:**
The state of health for lithium battery is necessary to ensure the reliability and safety for battery energy storage system. Accurate prediction battery state of health plays an extremely important role in guaranteeing safety and minimizing the maintenance costs. However, the complex physicochemical characteristics of battery degradation cannot be obtained directly. Here a novel Gaussian process regression model based on partial incremental capacity curve is proposed. First, an advanced Gaussian filter method is applied to obtain the smoothing incremental capacity curves. The health indexes are then extracted from the partial incremental capacity curves as the input features of the proposed model. Otherwise, the mean and the covariance function of the proposed method are applied to predict battery state of health and the model uncertainty, respectively. Four aging datasets from NASA data repository are employed for demonstrating the predictive capability and efficacy of the degradation model using the proposed method. Besides, different initial health conditions of the tested batteries are used to verify the robustness and reliability of the proposed method. Results show that the proposed method can provide accurate and robust state of health estimation.
       


### [Data-driven Prognostics with Predictive Uncertainty Estimation using Ensemble of Deep Ordinal Regression Models](https://arxiv.org/abs/1903.09795)

**Authors:**
Vishnu TV, Diksha, Pankaj Malhotra, Lovekesh Vig, Gautam Shroff

**Abstract:**
Prognostics or Remaining Useful Life (RUL) Estimation from multi-sensor time series data is useful to enable condition-based maintenance and ensure high operational availability of equipment. We propose a novel deep learning based approach for Prognostics with Uncertainty Quantification that is useful in scenarios where: (i) access to labeled failure data is scarce due to rarity of failures (ii) future operational conditions are unobserved and (iii) inherent noise is present in the sensor readings. All three scenarios mentioned are unavoidable sources of uncertainty in the RUL estimation process often resulting in unreliable RUL estimates. To address (i), we formulate RUL estimation as an Ordinal Regression (OR) problem, and propose LSTM-OR: deep Long Short Term Memory (LSTM) network based approach to learn the OR function. We show that LSTM-OR naturally allows for incorporation of censored operational instances in training along with the failed instances, leading to more robust learning. To address (ii), we propose a simple yet effective approach to quantify predictive uncertainty in the RUL estimation models by training an ensemble of LSTM-OR models. Through empirical evaluation on C-MAPSS turbofan engine benchmark datasets, we demonstrate that LSTM-OR is significantly better than the commonly used deep metric regression based approaches for RUL estimation, especially when failed training instances are scarce. Further, our uncertainty quantification approach yields high quality predictive uncertainty estimates while also leading to improved RUL estimates compared to single best LSTM-OR models.
       


## April
### [Tipping point analysis of electrical resistance data with early warning signals of failure for predictive maintenance](https://arxiv.org/abs/1904.04636)

**Authors:**
Valerie Livina, Adam Lewis, Martin Wickham

**Abstract:**
We apply tipping point analysis to measurements of electronic components commonly used in applications in the automotive or aviation industries and demonstrate early warning signals based on scaling properties of resistance time series. The analysis utilises the statistical physics framework with stochastic modelling by representing the measured time series as a composition of deterministic and stochastic components estimated from measurements. The early warning signals are observed much earlier than those estimated from conventional techniques, such as threshold-based failure detection, or bulk estimates used in Weibull failure analysis. The introduced techniques may be useful for predictive maintenance of power electronics, with industrial applications. We suggest that this approach can be applied to various electromagnetic measurements in power systems and energy applications.
       


### [Remaining Useful Life Estimation Using Functional Data Analysis](https://arxiv.org/abs/1904.06442)

**Authors:**
Qiyao Wang, Shuai Zheng, Ahmed Farahat, Susumu Serita, Chetan Gupta

**Abstract:**
Remaining Useful Life (RUL) of an equipment or one of its components is defined as the time left until the equipment or component reaches its end of useful life. Accurate RUL estimation is exceptionally beneficial to Predictive Maintenance, and Prognostics and Health Management (PHM). Data driven approaches which leverage the power of algorithms for RUL estimation using sensor and operational time series data are gaining popularity. Existing algorithms, such as linear regression, Convolutional Neural Network (CNN), Hidden Markov Models (HMMs), and Long Short-Term Memory (LSTM), have their own limitations for the RUL estimation task. In this work, we propose a novel Functional Data Analysis (FDA) method called functional Multilayer Perceptron (functional MLP) for RUL estimation. Functional MLP treats time series data from multiple equipment as a sample of random continuous processes over time. FDA explicitly incorporates both the correlations within the same equipment and the random variations across different equipment's sensor time series into the model. FDA also has the benefit of allowing the relationship between RUL and sensor variables to vary over time. We implement functional MLP on the benchmark NASA C-MAPSS data and evaluate the performance using two popularly-used metrics. Results show the superiority of our algorithm over all the other state-of-the-art methods.
       


### [Power System Dispatch with Marginal Degradation Cost of Battery Storage](https://arxiv.org/abs/1904.07771)

**Authors:**
Guannan He, Soummya Kar, Javad Mohammadi, Panayiotis Moutis, Jay F. Whitacre

**Abstract:**
Battery storage is essential for the future smart grid. The inevitable cell degradation renders the battery lifetime volatile and highly dependent on battery dispatch, and thus incurs opportunity cost. This paper rigorously derives the marginal degradation cost of battery for power system dispatch. The derived optimal marginal degradation cost is time-variant to reflect the time value of money and the functionality fade of battery and takes the form of a constant value divided by a discount factor plus a term related to battery state of health. In case studies, we demonstrate the evolution of the optimal marginal costs of degradation that corresponds to the optimal long-term dispatch outcome. We also show that the optimal marginal cost of degradation depends on the marginal cost of generation in the grid.
       


### [Optimal Policies for Recovery of Multiple Systems After Disruptions](https://arxiv.org/abs/1904.11615)

**Authors:**
Hemant Gehlot, Shreyas Sundaram, Satish V. Ukkusuri

**Abstract:**
We consider a scenario where a system experiences a disruption, and the states (representing health values) of its components continue to reduce over time, unless they are acted upon by a controller. Given this dynamical setting, we consider the problem of finding an optimal control (or switching) sequence to maximize the sum of the weights of the components whose states are brought back to the maximum value. We first provide several characteristics of the optimal policy for the general (fully heterogeneous) version of this problem. We then show that under certain conditions on the rates of repair and deterioration, we can explicitly characterize the optimal control policy as a function of the states. When the deterioration rate (when not being repaired) is larger than or equal to the repair rate, and the deterioration and repair rates as well as the weights are homogeneous across all the components, the optimal control policy is to target the component that has the largest state value at each time step. On the other hand, if the repair rates are sufficiently larger than the deterioration rates, the optimal control policy is to target the component whose state minus the deterioration rate is least in a particular subset of components at each time step.
       


### [Fault Diagnosis using Clustering. What Statistical Test to use for Hypothesis Testing?](https://arxiv.org/abs/1904.13365)

**Authors:**
Nagdev Amruthnath, Tarun Gupta

**Abstract:**
Predictive maintenance and condition-based monitoring systems have seen significant prominence in recent years to minimize the impact of machine downtime on production and its costs. Predictive maintenance involves using concepts of data mining, statistics, and machine learning to build models that are capable of performing early fault detection, diagnosing the faults and predicting the time to failure. Fault diagnosis has been one of the core areas where the actual failure mode of the machine is identified. In fluctuating environments such as manufacturing, clustering techniques have proved to be more reliable compared to supervised learning methods. One of the fundamental challenges of clustering is developing a test hypothesis and choosing an appropriate statistical test for hypothesis testing. Most statistical analyses use some underlying assumptions of the data which most real-world data is incapable of satisfying those assumptions. This paper is dedicated to overcoming the following challenge by developing a test hypothesis for fault diagnosis application using clustering technique and performing PERMANOVA test for hypothesis testing.
       


## May
### [A Neural Network-Evolutionary Computational Framework for Remaining Useful Life Estimation of Mechanical Systems](https://arxiv.org/abs/1905.05918)

**Authors:**
David Laredo, Zhaoyin Chen, Oliver Schütze, Jian-Qiao Sun

**Abstract:**
This paper presents a framework for estimating the remaining useful life (RUL) of mechanical systems. The framework consists of a multi-layer perceptron and an evolutionary algorithm for optimizing the data-related parameters. The framework makes use of a strided time window to estimate the RUL for mechanical components. Tuning the data-related parameters can become a very time consuming task. The framework presented here automatically reshapes the data such that the efficiency of the model is increased. Furthermore, the complexity of the model is kept low, e.g. neural networks with few hidden layers and few neurons at each layer. Having simple models has several advantages like short training times and the capacity of being in environments with limited computational resources such as embedded systems. The proposed method is evaluated on the publicly available C-MAPSS dataset, its accuracy is compared against other state-of-the art methods for the same dataset.
       


### [Rare Failure Prediction via Event Matching for Aerospace Applications](https://arxiv.org/abs/1905.11586)

**Author:**
Evgeny Burnaev

**Abstract:**
In this paper, we consider a problem of failure prediction in the context of predictive maintenance applications. We present a new approach for rare failures prediction, based on a general methodology, which takes into account peculiar properties of technical systems. We illustrate the applicability of the method on the real-world test cases from aircraft operations.
       


## July
### [Industrial DevOps](https://arxiv.org/abs/1907.01875)

**Authors:**
Wilhelm Hasselbring, Sören Henning, Björn Latte, Armin Möbius, Thomas Richter, Stefan Schalk, Maik Wojcieszak

**Abstract:**
The visions and ideas of Industry 4.0 require a profound interconnection of machines, plants, and IT systems in industrial production environments. This significantly increases the importance of software, which is coincidentally one of the main obstacles to the introduction of Industry 4.0. Lack of experience and knowledge, high investment and maintenance costs, as well as uncertainty about future developments cause many small and medium-sized enterprises hesitating to adopt Industry 4.0 solutions. We propose Industrial DevOps as an approach to introduce methods and culture of DevOps into industrial production environments. The fundamental concept of this approach is a continuous process of operation, observation, and development of the entire production environment. This way, all stakeholders, systems, and data can thus be integrated via incremental steps and adjustments can be made quickly. Furthermore, we present the Titan software platform accompanied by a role model for integrating production environments with Industrial DevOps. In two initial industrial application scenarios, we address the challenges of energy management and predictive maintenance with the methods, organizational structures, and tools of Industrial DevOps.
       


### [Forecasting remaining useful life: Interpretable deep learning approach via variational Bayesian inferences](https://arxiv.org/abs/1907.05146)

**Authors:**
Mathias Kraus, Stefan Feuerriegel

**Abstract:**
Predicting the remaining useful life of machinery, infrastructure, or other equipment can facilitate preemptive maintenance decisions, whereby a failure is prevented through timely repair or replacement. This allows for a better decision support by considering the anticipated time-to-failure and thus promises to reduce costs. Here a common baseline may be derived by fitting a probability density function to past lifetimes and then utilizing the (conditional) expected remaining useful life as a prognostic. This approach finds widespread use in practice because of its high explanatory power. A more accurate alternative is promised by machine learning, where forecasts incorporate deterioration processes and environmental variables through sensor data. However, machine learning largely functions as a black-box method and its forecasts thus forfeit most of the desired interpretability. As our primary contribution, we propose a structured-effect neural network for predicting the remaining useful life which combines the favorable properties of both approaches: its key innovation is that it offers both a high accountability and the flexibility of deep learning. The parameters are estimated via variational Bayesian inferences. The different approaches are compared based on the actual time-to-failure for aircraft engines. This demonstrates the performance and superior interpretability of our method, while we finally discuss implications for decision support.
       


### [Online Subspace Tracking for Damage Propagation Modeling and Predictive Analytics: Big Data Perspective](https://arxiv.org/abs/1907.11477)

**Author:**
Farhan Khan

**Abstract:**
We analyze damage propagation modeling of turbo-engines in a data-driven approach. We investigate subspace tracking assuming a low dimensional manifold structure and a static behavior during the healthy state of the machines. Our damage propagation model is based on the deviation of the data from the static behavior and uses the notion of health index as a measure of the condition. Hence, we incorporate condition-based maintenance and estimate the remaining useful life based on the current and previous health indexes. This paper proposes an algorithm that adapts well to the dynamics of the data and underlying system, and reduces the computational complexity by utilizing the low dimensional manifold structure of the data. A significant performance improvement is demonstrated over existing methods by using the proposed algorithm on CMAPSS Turbo-engine datasets.
       


### [Recurrent Neural Networks with Long Term Temporal Dependencies in Machine Tool Wear Diagnosis and Prognosis](https://arxiv.org/abs/1907.11848)

**Authors:**
Jianlei Zhang, Binil Starly

**Abstract:**
Data-driven approaches to automated machine condition monitoring are gaining popularity due to advancements made in sensing technologies and computing algorithms. This paper proposes the use of a deep learning model, based on Long Short-Term Memory (LSTM) architecture for a recurrent neural network (RNN) which captures long term dependencies for modeling sequential data. In the context of estimating cutting tool wear amounts, this LSTM based RNN approach utilizes a system transition and system observation function based on a minimally intrusive vibration sensor signal located near the workpiece fixtures. By applying an LSTM based RNN, the method helps to avoid building an analytic model for specific tool wear machine degradation, overcoming the assumptions made by Hidden Markov Models, Kalman filter, and Particle filter based approaches. The proposed approach is tested using experiments performed on a milling machine. We have demonstrated one-step and two-step look ahead cutting tool state prediction using online indirect measurements obtained from vibration signals. Additionally, the study also estimates remaining useful life (RUL) of a machine cutting tool insert through generative RNN. The experimental results show that our approach, applying the LSTM to model system observation and transition function is able to outperform the functions modeled with a simple RNN.
       


## August
### [A Dynamic Analysis of Energy Storage with Renewable and Diesel Generation using Volterra Equations](https://arxiv.org/abs/1908.01310)

**Authors:**
Denis Sidorov, Ildar Muftahov, Nikita Tomin, Dmitriy Karamov, Daniil Panasetsky, Aliona Dreglea, Fang Liu, Aoife Foley

**Abstract:**
Energy storage systems will play a key role in the power system of the twenty first century considering the large penetrations of variable renewable energy, growth in transport electrification and decentralisation of heating loads. Therefore reliable real time methods to optimise energy storage, demand response and generation are vital for power system operations. This paper presents a concise review of battery energy storage and an example of battery modelling for renewable energy applications and second details an adaptive approach to solve this load levelling problem with storage. A dynamic evolutionary model based on the first kind Volterra integral equation is used in both cases. A direct regularised numerical method is employed to find the least-cost dispatch of the battery in terms of integral equation solution. Validation on real data shows that the proposed evolutionary Volterra model effectively generalises conventional discrete integral model taking into account both state of health and the availability of generation/storage.
       


### [A Condition Monitoring Concept Studied at the MST Prototype for the Cherenkov Telescope Array](https://arxiv.org/abs/1908.02180)

**Authors:**
Victor Barbosa Martins, Markus Garczarczyk, Gerrit Spengler, Ullrich Schwanke

**Abstract:**
The Cherenkov Telescope Array (CTA) is a future ground-based gamma-ray observatory that will provide unprecedented sensitivity and angular resolution for the detection of gamma rays with energies above a few tens of GeV. In comparison to existing instruments (like H.E.S.S., MAGIC, and VERITAS) the sensitivity will be improved by installing two extended arrays of telescopes in the northern and southern hemisphere, respectively. A large number of planned telescopes (>100 in total) motivates the application of predictive maintenance techniques to the individual telescopes. A constant and automatic condition monitoring of the mechanical telescope structure and of the drive system (motors, gears) is considered for this purpose. The condition monitoring system aims at detecting degradations well before critical errors occur; it should help to ensure long-term operation and to reduce the maintenance efforts of the observatory. We present approaches for the condition monitoring of the structure and the drive system of Medium-Sized Telescopes (MSTs), respectively. The overall concept has been developed and tested at the MST prototype for CTA in Berlin. The sensors used, the joint data acquisition system, possible analysis methods (like Operational Modal Analysis, OMA, and Experimental Modal Analysis, EMA) and first performance results are discussed.
       


### [Machine Learning and the Internet of Things Enable Steam Flood Optimization for Improved Oil Production](https://arxiv.org/abs/1908.11319)

**Authors:**
Mi Yan, Jonathan C. MacDonald, Chris T. Reaume, Wesley Cobb, Tamas Toth, Sarah S. Karthigan

**Abstract:**
Recently developed machine learning techniques, in association with the Internet of Things (IoT) allow for the implementation of a method of increasing oil production from heavy-oil wells. Steam flood injection, a widely used enhanced oil recovery technique, uses thermal and gravitational potential to mobilize and dilute heavy oil in situ to increase oil production. In contrast to traditional steam flood simulations based on principles of classic physics, we introduce here an approach using cutting-edge machine learning techniques that have the potential to provide a better way to describe the performance of steam flood. We propose a workflow to address a category of time-series data that can be analyzed with supervised machine learning algorithms and IoT. We demonstrate the effectiveness of the technique for forecasting oil production in steam flood scenarios. Moreover, we build an optimization system that recommends an optimal steam allocation plan, and show that it leads to a 3% improvement in oil production. We develop a minimum viable product on a cloud platform that can implement real-time data collection, transfer, and storage, as well as the training and implementation of a cloud-based machine learning model. This workflow also offers an applicable solution to other problems with similar time-series data structures, like predictive maintenance.
       


## September
### [Artificial Neural Networks and Adaptive Neuro-fuzzy Models for Prediction of Remaining Useful Life](https://arxiv.org/abs/1909.02115)

**Authors:**
Razieh Tavakoli, Mohammad Najafi, Ali Sharifara

**Abstract:**
The U.S. water distribution system contains thousands of miles of pipes constructed from different materials, and of various sizes, and age. These pipes suffer from physical, environmental, structural and operational stresses, causing deterioration which eventually leads to their failure. Pipe deterioration results in increased break rates, reduced hydraulic capacity, and detrimental impacts on water quality. Therefore, it is crucial to use accurate models to forecast deterioration rates along with estimating the remaining useful life of the pipes to implement essential interference plans in order to prevent catastrophic failures. This paper discusses a computational model that forecasts the RUL of water pipes by applying Artificial Neural Networks (ANNs) as well as Adaptive Neural Fuzzy Inference System (ANFIS). These models are trained and tested acquired field data to identify the significant parameters that impact the prediction of RUL. It is concluded that, on average, with approximately 10\% of wall thickness loss in existing cast iron, ductile iron, asbestos-cement, and steel water pipes, the reduction of the remaining useful life is approximately 50%
       


### [Transfer learning for Remaining Useful Life Prediction Based on Consensus Self-Organizing Models](https://arxiv.org/abs/1909.07053)

**Authors:**
Yuantao Fan, Sławomir Nowaczyk, Thorsteinn Rögnvaldsson

**Abstract:**
The traditional paradigm for developing machine prognostics usually relies on generalization from data acquired in experiments under controlled conditions prior to deployment of the equipment. Detecting or predicting failures and estimating machine health in this way assumes that future field data will have a very similar distribution to the experiment data. However, many complex machines operate under dynamic environmental conditions and are used in many different ways. This makes collecting comprehensive data very challenging, and the assumption that pre-deployment data and post-deployment data follow very similar distributions is unlikely to hold. Transfer Learning (TL) refers to methods for transferring knowledge learned in one setting (the source domain) to another setting (the target domain). In this work, we present a TL method for predicting Remaining Useful Life (RUL) of equipment, under the assumption that labels are available only for the source domain and not the target domain. This setting corresponds to generalizing from a limited number of run-to-failure experiments performed prior to deployment into making prognostics with data coming from deployed equipment that is being used under multiple new operating conditions and experiencing previously unseen faults. We employ a deviation detection method, Consensus Self-Organizing Models (COSMO), to create transferable features for building the RUL regression model. These features capture how different target equipment is in comparison to its peers. The efficiency of the proposed TL method is demonstrated using the NASA Turbofan Engine Degradation Simulation Data Set. Models using the COSMO transferable features show better performance than other methods on predicting RUL when the target domain is more complex than the source domain.
       


### [Simultaneous Identification and Control Using Active Signal Injection for Series Hybrid Electric Vehicles based on Dynamic Programming](https://arxiv.org/abs/1909.08062)

**Authors:**
Haojie Zhu, Ziyou Song, Jun Hou, Heath Hofmann, Jing Sun

**Abstract:**
Hybrid electric vehicles (HEVs) have an over-actuated system by including two power sources, a battery pack and an internal combustion engine. This feature of HEV is exploited in this paper to simultaneously achieve accurate identification of battery parameters/states. By actively injecting current signals, state of charge, state of health, and other battery parameters can be estimated in a specific sequence to improve the identification performance when compared to the case where all parameters and states are estimated concurrently using the baseline current signals. A dynamic programming strategy is developed to provide the benchmark results about how to balance the conflicting objectives corresponding to identification and system efficiency. The tradeoff between different objectives is presented to optimize the current profile so that the richness of signal can be ensured and the fuel economy can be optimized. In addition, simulation results show that the Root-Mean-Square error of the estimation can be decreased by up to 100% at a cost of less than 2% increase in fuel consumption. With the proposed simultaneous identification and control algorithm, the parameters/states of the battery can be monitored to ensure safe and efficient application of the battery for HEVs.
       


### [A deep adversarial approach based on multi-sensor fusion for remaining useful life prognostics](https://arxiv.org/abs/1909.10246)

**Authors:**
David Verstraete, Enrique Droguett, Mohammad Modarres

**Abstract:**
Multi-sensor systems are proliferating the asset management industry and by proxy, the structural health management community. Asset managers are beginning to require a prognostics and health management system to predict and assess maintenance decisions. These systems handle big machinery data and multi-sensor fusion and integrate remaining useful life prognostic capabilities. We introduce a deep adversarial learning approach to damage prognostics. A non-Markovian variational inference-based model incorporating an adversarial training algorithm framework was developed. The proposed framework was applied to a public multi-sensor data set of turbofan engines to demonstrate its ability to predict remaining useful life. We find that using the deep adversarial based approach results in higher performing remaining useful life predictions.
       


## October
### [False Data Injection Attacks in Internet of Things and Deep Learning enabled Predictive Analytics](https://arxiv.org/abs/1910.01716)

**Authors:**
Gautam Raj Mode, Prasad Calyam, Khaza Anuarul Hoque

**Abstract:**
Industry 4.0 is the latest industrial revolution primarily merging automation with advanced manufacturing to reduce direct human effort and resources. Predictive maintenance (PdM) is an industry 4.0 solution, which facilitates predicting faults in a component or a system powered by state-of-the-art machine learning (ML) algorithms and the Internet-of-Things (IoT) sensors. However, IoT sensors and deep learning (DL) algorithms, both are known for their vulnerabilities to cyber-attacks. In the context of PdM systems, such attacks can have catastrophic consequences as they are hard to detect due to the nature of the attack. To date, the majority of the published literature focuses on the accuracy of DL enabled PdM systems and often ignores the effect of such attacks. In this paper, we demonstrate the effect of IoT sensor attacks on a PdM system. At first, we use three state-of-the-art DL algorithms, specifically, Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Convolutional Neural Network (CNN) for predicting the Remaining Useful Life (RUL) of a turbofan engine using NASA's C-MAPSS dataset. The obtained results show that the GRU-based PdM model outperforms some of the recent literature on RUL prediction using the C-MAPSS dataset. Afterward, we model two different types of false data injection attacks (FDIA) on turbofan engine sensor data and evaluate their impact on CNN, LSTM, and GRU-based PdM systems. The obtained results demonstrate that FDI attacks on even a few IoT sensors can strongly defect the RUL prediction. However, the GRU-based PdM model performs better in terms of accuracy and resiliency. Lastly, we perform a study on the GRU-based PdM model using four different GRU networks with different sequence lengths. Our experiments reveal an interesting relationship between the accuracy, resiliency and sequence length for the GRU-based PdM models.
       


### [The false myth of the rise in self-citations, and the impressively positive effect of bibliometric evaluations on the increase of the impact of Italian research](https://arxiv.org/abs/1910.02948)

**Authors:**
Pietro D'Antuono, Michele Ciavarella

**Abstract:**
It has recently been claimed by Baccini and coauthors that due to ANVUR's bibliometric evaluations of individuals, departments, and universities, in Italy there has been a surge in self-citations in the last ten years, thus increasing the "inwardness" of Italian research more than has happened abroad. We have studied the database of Ioannidis et al. published on 12 August 2019 of the one hundred thousand most "highly cited" scientists, including about two thousand Italians, and we found that the problem of self-citations in relation to this scientific elite is not significant in Italy, while perhaps observing a small deviation in the low scores in the rankings. The effect indicated by Baccini et al. consequently, does not seem worrying for the scientific elite (we quantified it in 2 percent of the total of scientists of the "best" one hundred thousand), and is probably largely concentrated in the further less cited scientists. Evaluation agencies like ANVUR should probably exclude self-citations in future evaluations, for the noise introduced by the young researchers. The overall state of health of the Italian research system and the positive effect of the ANVUR assessments are demonstrated by the number of Italian researchers in the top one hundred thousand, which has increased by comparing the "career" databased of 22 years, with that of the "young" researchers in the "2017" database. Italy, looking at the elite researchers, not only is not the most indulgent in self-citations, but has shown the best improvements, proving that the introduction of ANVUR had a positive effect. Indeed, all countries apart from Italy have suffered a decline, even substantial (-20 percent on a national Japan scale), of the number of researchers present in the 2017 data sets compared to career data. Italy instead shows a +0.2 percent on a global basis and an impressive +11.53 percent on a national basis.
       


### [Network Scanning and Mapping for IIoT Edge Node Device Security](https://arxiv.org/abs/1910.07622)

**Authors:**
Matthias Niedermaier, Florian Fischer, Dominik Merli, Georg Sigl

**Abstract:**
The amount of connected devices in the industrial environment is growing continuously, due to the ongoing demands of new features like predictive maintenance. New business models require more data, collected by IIoT edge node sensors based on inexpensive and low performance Microcontroller Units (MCUs). A negative side effect of this rise of interconnections is the increased attack surface, enabled by a larger network with more network services. Attaching badly documented and cheap devices to industrial networks often without permission of the administrator even further increases the security risk. A decent method to monitor the network and detect "unwanted" devices is network scanning. Typically, this scanning procedure is executed by a computer or server in each sub-network. In this paper, we introduce network scanning and mapping as a building block to scan directly from the Industrial Internet of Things (IIoT) edge node devices. This module scans the network in a pseudo-random periodic manner to discover devices and detect changes in the network structure. Furthermore, we validate our approach in an industrial testbed to show the feasibility of this approach.
       


### [A Scalable Predictive Maintenance Model for Detecting Wind Turbine Component Failures Based on SCADA Data](https://arxiv.org/abs/1910.09808)

**Authors:**
Lorenzo Gigoni, Alessandro Betti, Mauro Tucci, Emanuele Crisostomi

**Abstract:**
In this work, a novel predictive maintenance system is presented and applied to the main components of wind turbines. The proposed model is based on machine learning and statistical process control tools applied to SCADA (Supervisory Control And Data Acquisition) data of critical components. The test campaign was divided into two stages: a first two years long offline test, and a second one year long real-time test. The offline test used historical faults from six wind farms located in Italy and Romania, corresponding to a total of 150 wind turbines and an overall installed nominal power of 283 MW. The results demonstrate outstanding capabilities of anomaly prediction up to 2 months before device unscheduled downtime. Furthermore, the real-time 12-months test confirms the ability of the proposed system to detect several anomalies, therefore allowing the operators to identify the root causes, and to schedule maintenance actions before reaching a catastrophic stage.
       


### [Software Framework for Tribotronic Systems](https://arxiv.org/abs/1910.13764)

**Authors:**
Jarno Kansanaho, Tommi Kärkkäinen

**Abstract:**
Increasing the capabilities of sensors and computer algorithms produces a need for structural support that would solve recurring problems. Autonomous tribotronic systems self-regulate based on feedback acquired from interacting surfaces in relative motion. This paper describes a software framework for tribotronic systems. An example of such an application is a rolling element bearing (REB) installation with a vibration sensor. The presented plug-in framework offers functionalities for vibration data management, feature extraction, fault detection, and remaining useful life (RUL) estimation. The framework was tested using bearing vibration data acquired from NASA's prognostics data repository, and the evaluation included a run-through from feature extraction to fault detection to remaining useful life estimation. The plug-in implementations are easy to update and new implementations are easily deployable, even in run-time. The proposed software framework improves the performance, efficiency, and reliability of a tribotronic system. In addition, the framework facilitates the evaluation of the configuration complexity of the plug-in implementation.
       


## November
### [Condition monitoring and early diagnostics methodologies for hydropower plants](https://arxiv.org/abs/1911.06242)

**Authors:**
Alessandro Betti, Emanuele Crisostomi, Gianluca Paolinelli, Antonio Piazzi, Fabrizio Ruffini, Mauro Tucci

**Abstract:**
Hydropower plants are one of the most convenient option for power generation, as they generate energy exploiting a renewable source, they have relatively low operating and maintenance costs, and they may be used to provide ancillary services, exploiting the large reservoirs of available water. The recent advances in Information and Communication Technologies (ICT) and in machine learning methodologies are seen as fundamental enablers to upgrade and modernize the current operation of most hydropower plants, in terms of condition monitoring, early diagnostics and eventually predictive maintenance. While very few works, or running technologies, have been documented so far for the hydro case, in this paper we propose a novel Key Performance Indicator (KPI) that we have recently developed and tested on operating hydropower plants. In particular, we show that after more than one year of operation it has been able to identify several faults, and to support the operation and maintenance tasks of plant operators. Also, we show that the proposed KPI outperforms conventional multivariable process control charts, like the Hotelling $t_2$ index.
       


### [A Comparative Study between Bayesian and Frequentist Neural Networks for Remaining Useful Life Estimation in Condition-Based Maintenance](https://arxiv.org/abs/1911.06256)

**Author:**
Luca Della Libera

**Abstract:**
In the last decade, deep learning (DL) has outperformed model-based and statistical approaches in predicting the remaining useful life (RUL) of machinery in the context of condition-based maintenance. One of the major drawbacks of DL is that it heavily depends on a large amount of labeled data, which are typically expensive and time-consuming to obtain, especially in industrial applications. Scarce training data lead to uncertain estimates of the model's parameters, which in turn result in poor prognostic performance. Quantifying this parameter uncertainty is important in order to determine how reliable the prediction is. Traditional DL techniques such as neural networks are incapable of capturing the uncertainty in the training data, thus they are overconfident about their estimates. On the contrary, Bayesian deep learning has recently emerged as a promising solution to account for uncertainty in the training process, achieving state-of-the-art performance in many classification and regression tasks. In this work Bayesian DL techniques such as Bayesian dense neural networks and Bayesian convolutional neural networks are applied to RUL estimation and compared to their frequentist counterparts from the literature. The effectiveness of the proposed models is verified on the popular C-MAPSS dataset. Furthermore, parameter uncertainty is quantified and used to gain additional insight into the data.
       


## December
### [An Attribute Oriented Induction based Methodology for Data Driven Predictive Maintenance](https://arxiv.org/abs/1912.00662)

**Authors:**
Javier Fernandez-Anakabe, Ekhi Zugasti Uriguen, Urko Zurutuza Ortega

**Abstract:**
Attribute Oriented Induction (AOI) is a data mining algorithm used for extracting knowledge of relational data, taking into account expert knowledge. It is a clustering algorithm that works by transforming the values of the attributes and converting an instance into others that are more generic or ambiguous. In this way, it seeks similarities between elements to generate data groupings. AOI was initially conceived as an algorithm for knowledge discovery in databases, but over the years it has been applied to other areas such as spatial patterns, intrusion detection or strategy making. In this paper, AOI has been extended to the field of Predictive Maintenance. The objective is to demonstrate that combining expert knowledge and data collected from the machine can provide good results in the Predictive Maintenance of industrial assets. To this end we adapted the algorithm and used an LSTM approach to perform both the Anomaly Detection (AD) and the Remaining Useful Life (RUL). The results obtained confirm the validity of the proposal, as the methodology was able to detect anomalies, and calculate the RUL until breakage with considerable degree of accuracy.
       


### [ADEPOS: A Novel Approximate Computing Framework for Anomaly Detection Systems and its Implementation in 65nm CMOS](https://arxiv.org/abs/1912.01853)

**Authors:**
Sumon Kumar Bose, Bapi Kar, Mohendra Roy, Pradeep Kumar Gopalakrishnan, Zhang Lei, Aakash Patil, Arindam Basu

**Abstract:**
To overcome the energy and bandwidth limitations of traditional IoT systems, edge computing or information extraction at the sensor node has become popular. However, now it is important to create very low energy information extraction or pattern recognition systems. In this paper, we present an approximate computing method to reduce the computation energy of a specific type of IoT system used for anomaly detection (e.g. in predictive maintenance, epileptic seizure detection, etc). Termed as Anomaly Detection Based Power Savings (ADEPOS), our proposed method uses low precision computing and low complexity neural networks at the beginning when it is easy to distinguish healthy data. However, on the detection of anomalies, the complexity of the network and computing precision are adaptively increased for accurate predictions. We show that ensemble approaches are well suited for adaptively changing network size. To validate our proposed scheme, a chip has been fabricated in UMC65nm process that includes an MSP430 microprocessor along with an on-chip switching mode DC-DC converter for dynamic voltage and frequency scaling. Using NASA bearing dataset for machine health monitoring, we show that using ADEPOS we can achieve 8.95X saving of energy along the lifetime without losing any detection accuracy. The energy savings are obtained by reducing the execution time of the neural network on the microprocessor.
       


### [Low computational cost method for online parameter identification of Li-ion battery in battery management systems using matrix condition number](https://arxiv.org/abs/1912.02600)

**Authors:**
Minho Kim, Kwangrae Kim, Soohee Han

**Abstract:**
Monitoring the state of health for Li-ion batteries is crucial in the battery management system (BMS), which helps end-users use batteries efficiently and safely. Battery state of health can be monitored by identifying parameters of battery models using various algorithms. Due to the low computation power of BMS and time-varying parameters, it is very important to develop an online algorithm with low computational cost. Among various methods, Equivalent circuit model (ECM) -based recursive least squares (RLS) parameter identification is well suited for such difficult BMS environments. However, one well-known critical problem of RLS is that it is very likely to be numerically unstable unless the measured inputs make enough excitation of the battery models. In this work, A new version of RLS, which is called condition memory recursive least squares (CMRLS) is developed for the Li-ion battery parameter identification to solve such problems and to take advantage of RLS at the same time by varying forgetting factor according to condition numbers. In CMRLS, exact condition numbers are monitored with simple computations using recursive relations between RLS variables. The performance of CMRLS is compared with the original RLS through Li-ion battery simulations. It is shown that CMRLS identifies Li-ion battery parameters about 100 times accurately than RLS in terms of mean absolute error.
       


### [Survey of prognostics methods for condition-based maintenance in engineering systems](https://arxiv.org/abs/1912.02708)

**Authors:**
Ehsan Taheri, Ilya Kolmanovsky, Oleg Gusikhin

**Abstract:**
It is not surprising that the idea of efficient maintenance algorithms (originally motivated by strict emission regulations, and now driven by safety issues, logistics and customer satisfaction) has culminated in the so-called condition-based maintenance program. Condition-based program/monitoring consists of two major tasks, i.e., \textit{diagnostics} and \textit{prognostics} each of which has provided the impetus and technical challenges to the scientists and engineers in various fields of engineering. Prognostics deals with the prediction of the remaining useful life, future condition, or probability of reliable operation of an equipment based on the acquired condition monitoring data. This approach to modern maintenance practice promises to reduce the downtime, spares inventory, maintenance costs, and safety hazards. Given the significance of prognostics capabilities and the maturity of condition monitoring technology, there have been an increasing number of publications on machinery prognostics in the past few years. These publications cover a wide range of issues important to prognostics. Fortunately, improvement in computational resources technology has come to the aid of engineers by presenting more powerful onboard computational resources to make some aspects of these new problems tractable. In addition, it is possible to even leverage connected vehicle information through cloud-computing. Our goal is to review the state of the art and to summarize some of the recent advances in prognostics with the emphasis on models, algorithms and technologies used for data processing and decision making.
       


### [Parameters inference and model reduction for the Single-Particle Model of Li ion cells](https://arxiv.org/abs/1912.05807)

**Authors:**
Michael Khasin, Chetan S. Kulkarni, Kai Goebel

**Abstract:**
The Single-Particle Model (SPM) of Li ion cell \cite{Santhanagopalan06, Guo2011} is a computationally efficient and fairly accurate model for simulating Li ion cell cycling behavior at weak to moderate currents. The model depends on a large number of parameters describing the geometry and material properties of a cell components. In order to use the SPM for simulation of a 18650 LP battery cycling behavior, we fitted the values of the model parameters to a cycling data. We found that the distribution of parametric values for which the SPM fits the data accurately is strongly delocalized in the (nondimensionalized) parametric space, with variances in certain directions larger by many orders of magnitude than in other directions.
  This property of the SPM is known to be shared by a multitude of the so-called "sloppy models" \cite{Brown2003, Waterfall2006}, characterized by a few stiff directions in the parametric space, in which the predicted behavior varies significantly, and a number of sloppy directions in which the behavior doesn't change appreciably. As a consequence, only stiff parameters of the SPM can be inferred with a fair degree of certainty and these are the parameters which determine the cycling behavior of the battery. Based on geometrical insights from the Sloppy Models theory, we derive an hierarchy of reduced models for the SPM. The fully reduced model depends on only three stiff effective parameters which can be used for the battery state of health characterization.
       


### [A Survey of Predictive Maintenance: Systems, Purposes and Approaches](https://arxiv.org/abs/1912.07383)

**Authors:**
Tianwen Zhu, Yongyi Ran, Xin Zhou, Yonggang Wen

**Abstract:**
This paper highlights the importance of maintenance techniques in the coming industrial revolution, reviews the evolution of maintenance techniques, and presents a comprehensive literature review on the latest advancement of maintenance techniques, i.e., Predictive Maintenance (PdM), with emphasis on system architectures, optimization objectives, and optimization methods. In industry, any outages and unplanned downtime of machines or systems would degrade or interrupt a company's core business, potentially resulting in significant penalties and immeasurable reputation and economic loss. Existing traditional maintenance approaches, such as Reactive Maintenance (RM) and Preventive Maintenance (PM), suffer from high prevent and repair costs, inadequate or inaccurate mathematical degradation processes, and manual feature extraction. The incoming fourth industrial revolution is also demanding for a new maintenance paradigm to reduce the maintenance cost and downtime, and increase system availability and reliability. Predictive Maintenance (PdM) is envisioned the solution. In this survey, we first provide a high-level view of the PdM system architectures including PdM 4.0, Open System Architecture for Condition Based Monitoring (OSA-CBM), and cloud-enhanced PdM system. Then, we review the specific optimization objectives, which mainly comprise cost minimization, availability/reliability maximization, and multi-objective optimization. Furthermore, we present the optimization methods to achieve the aforementioned objectives, which include traditional Machine Learning (ML) based and Deep Learning (DL) based approaches. Finally, we highlight the future research directions that are critical to promote the application of DL techniques in the context of PdM.
       


# 2020
## January
### [Infrequent adverse event prediction in low carbon energy production using machine learning](https://arxiv.org/abs/2001.06916)

**Authors:**
Stefano Coniglio, Anthony J. Dunn, Alain B. Zemkoho

**Abstract:**
We address the problem of predicting the occurrence of infrequent adverse events in the context of predictive maintenance. We cast the corresponding machine learning task as an imbalanced classification problem and propose a framework for solving it that is capable of leveraging different classifiers in order to predict the occurrence of an adverse event before it takes place. In particular, we focus on two applications arising in low-carbon energy production: foam formation in anaerobic digestion and condenser tube leakage in the steam turbines of a nuclear power station. The results of an extensive set of omputational experiments show the effectiveness of the techniques that we propose.
       


### [Multi-label Prediction in Time Series Data using Deep Neural Networks](https://arxiv.org/abs/2001.10098)

**Authors:**
Wenyu Zhang, Devesh K. Jha, Emil Laftchiev, Daniel Nikovski

**Abstract:**
This paper addresses a multi-label predictive fault classification problem for multidimensional time-series data. While fault (event) detection problems have been thoroughly studied in literature, most of the state-of-the-art techniques can't reliably predict faults (events) over a desired future horizon. In the most general setting of these types of problems, one or more samples of data across multiple time series can be assigned several concurrent fault labels from a finite, known set and the task is to predict the possibility of fault occurrence over a desired time horizon. This type of problem is usually accompanied by strong class imbalances where some classes are represented by only a few samples. Importantly, in many applications of the problem such as fault prediction and predictive maintenance, it is exactly these rare classes that are of most interest. To address the problem, this paper proposes a general approach that utilizes a multi-label recurrent neural network with a new cost function that accentuates learning in the imbalanced classes. The proposed algorithm is tested on two public benchmark datasets: an industrial plant dataset from the PHM Society Data Challenge, and a human activity recognition dataset. The results are compared with state-of-the-art techniques for time-series classification and evaluation is performed using the F1-score, precision and recall.
       


## February
### [Health Assessment and Prognostics Based on Higher Order Hidden Semi-Markov Models](https://arxiv.org/abs/2002.05272)

**Authors:**
Ying Liao, Yisha Xiang, Min Wang

**Abstract:**
This paper presents a new and flexible prognostics framework based on a higher order hidden semi-Markov model (HOHSMM) for systems or components with unobservable health states and complex transition dynamics. The HOHSMM extends the basic hidden Markov model (HMM) by allowing the hidden state to depend on its more distant history and assuming generally distributed state duration. An effective Gibbs sampling algorithm is designed for statistical inference of an HOHSMM. The performance of the proposed HOHSMM sampler is evaluated by conducting a simulation experiment. We further design a decoding algorithm to estimate the hidden health states using the learned model. Remaining useful life (RUL) is predicted using a simulation approach given the decoded hidden states. The practical utility of the proposed prognostics framework is demonstrated by a case study on NASA turbofan engines. The results show that the HOHSMM-based prognostics framework provides good hidden health state assessment and RUL estimation for complex systems.
       


### [A Survey on Predictive Maintenance for Industry 4.0](https://arxiv.org/abs/2002.08224)

**Authors:**
Christian Krupitzer, Tim Wagenhals, Marwin Züfle, Veronika Lesch, Dominik Schäfer, Amin Mozaffarin, Janick Edinger, Christian Becker, Samuel Kounev

**Abstract:**
Production issues at Volkswagen in 2016 lead to dramatic losses in sales of up to 400 million Euros per week. This example shows the huge financial impact of a working production facility for companies. Especially in the data-driven domains of Industry 4.0 and Industrial IoT with intelligent, connected machines, a conventional, static maintenance schedule seems to be old-fashioned. In this paper, we present a survey on the current state of the art in predictive maintenance for Industry 4.0. Based on a structured literate survey, we present a classification of predictive maintenance in the context of Industry 4.0 and discuss recent developments in this area.
       


### [Estimation of conditional mixture Weibull distribution with right-censored data using neural network for time-to-event analysis](https://arxiv.org/abs/2002.09358)

**Authors:**
Achraf Bennis, Sandrine Mouysset, Mathieu Serrurier

**Abstract:**
In this paper, we consider survival analysis with right-censored data which is a common situation in predictive maintenance and health field. We propose a model based on the estimation of two-parameter Weibull distribution conditionally to the features. To achieve this result, we describe a neural network architecture and the associated loss functions that takes into account the right-censored data. We extend the approach to a finite mixture of two-parameter Weibull distributions. We first validate that our model is able to precisely estimate the right parameters of the conditional Weibull distribution on synthetic datasets. In numerical experiments on two real-word datasets (METABRIC and SEER), our model outperforms the state-of-the-art methods. We also demonstrate that our approach can consider any survival time horizon.
       


### [Driving with Data in the Motor City: Mining and Modeling Vehicle Fleet Maintenance Data](https://arxiv.org/abs/2002.10010)

**Authors:**
Josh Gardner, Jawad Mroueh, Natalia Jenuwine, Noah Weaverdyck, Samuel Krassenstein, Arya Farahi, Danai Koutra

**Abstract:**
The City of Detroit maintains an active fleet of over 2500 vehicles, spending an annual average of over \$5 million on purchases and over \$7.7 million on maintenance. Modeling patterns and trends in this data is of particular importance to a variety of stakeholders, particularly as Detroit emerges from Chapter 9 bankruptcy, but the structure in such data is complex, and the city lacks dedicated resources for in-depth analysis. The City of Detroit's Operations and Infrastructure Group and the University of Michigan initiated a collaboration which seeks to address this unmet need by analyzing data from the City of Detroit's vehicle fleet. This work presents a case study and provides the first data-driven benchmark, demonstrating a suite of methods to aid in data understanding and prediction for large vehicle maintenance datasets. We present analyses to address three key questions raised by the stakeholders, related to discovering multivariate maintenance patterns over time; predicting maintenance; and predicting vehicle- and fleet-level costs. We present a novel algorithm, PRISM, for automating multivariate sequential data analyses using tensor decomposition. This work is a first of its kind that presents both methodologies and insights to guide future civic data research.
       


### [Multi-agent maintenance scheduling based on the coordination between central operator and decentralized producers in an electricity market](https://arxiv.org/abs/2002.12217)

**Authors:**
Pegah Rokhforoz, Blazhe Gjorgiev, Giovanni Sansavini, Olga Fink

**Abstract:**
Condition-based and predictive maintenance enable early detection of critical system conditions and thereby enable decision makers to forestall faults and mitigate them. However, decision makers also need to take the operational and production needs into consideration for optimal decision-making when scheduling maintenance activities. Particularly in network systems, such as power grids, decisions on the maintenance of single assets can affect the entire network and are, therefore, more complex. This paper proposes a two-level multi-agent decision support systems for the generation maintenance decision (GMS) of power grids in an electricity markets. The aim of the GMS is to minimize the generation cost while maximizing the system reliability. The proposed framework integrates a central coordination system, i.e. the transmission system operator (TSO), and distributed agents representing power generation units that act to maximize their profit and decide about the optimal maintenance time slots while ensuring the fulfilment of the energy demand. The objective function of agents (power generation companies) is based on the reward and the penalty that they obtain from the interplay between power production and loss of production due to failure, respectively. The optimal strategy of agents is then derived using a distributed algorithm, where agents choose their optimal maintenance decision and send their decisions to the central coordinating system. The TSO decides whether to accept the agents' decisions by considering the market reliability aspects and power supply constraints. To solve this coordination problem, we propose a negotiation algorithm using an incentive signal to coordinate the agents' and central system's decisions such that all the agents' decisions can be accepted by the central system. We demonstrate the efficiency of our proposed algorithm using a IEEE 39 bus system.
       


## March
### [NVMe and PCIe SSD Monitoring in Hyperscale Data Centers](https://arxiv.org/abs/2003.11267)

**Authors:**
Nikhil Khatri, Shirshendu Chakrabarti

**Abstract:**
With low latency, high throughput and enterprise-grade reliability, SSDs have become the de-facto choice for storage in the data center. As a result, SSDs are used in all online data stores in LinkedIn. These apps persist and serve critical user data and have millisecond latencies. For the hosts serving these applications, SSD faults are the single largest cause of failure. Frequent SSD failures result in significant downtime for critical applications. They also generate a significant downstream RCA (Root Cause Analysis) load for systems operations teams. A lack of insight into the runtime characteristics of these drives results in limited ability to provide accurate RCAs for such issues and hinders the ability to provide credible, long term fixes to such issues. In this paper we describe the system developed at LinkedIn to facilitate the real-time monitoring of SSDs and the insights we gained into failure characteristics. We describe how we used that insight to perform predictive maintenance and present the resulting reduction of man-hours spent on maintenance.
       


## April
### [Diversity-Aware Weighted Majority Vote Classifier for Imbalanced Data](https://arxiv.org/abs/2004.07605)

**Authors:**
Anil Goyal, Jihed Khiari

**Abstract:**
In this paper, we propose a diversity-aware ensemble learning based algorithm, referred to as DAMVI, to deal with imbalanced binary classification tasks. Specifically, after learning base classifiers, the algorithm i) increases the weights of positive examples (minority class) which are "hard" to classify with uniformly weighted base classifiers; and ii) then learns weights over base classifiers by optimizing the PAC-Bayesian C-Bound that takes into account the accuracy and diversity between the classifiers. We show efficiency of the proposed approach with respect to state-of-art models on predictive maintenance task, credit card fraud detection, webpage classification and medical applications.
       


### [Thermal Accelerated Aging Methods for Magnet Wire: A Review](https://arxiv.org/abs/2004.09187)

**Authors:**
Lukas L. Korcak, Darren F. Kavanagh

**Abstract:**
This paper focuses on accelerated aging methods for magnet wire. Reliability of electrical devices such as coils, motors, relays, solenoids and transformers is heavily dependent on the Electrical Insulation System (EIS). Accelerated aging methods are used to rapidly simulate the conditions in real life, which is typically years (20,000 hours) depending on the operating conditions. The purpose of accelerated aging is to bring lifetime of an EIS to hours, days or weeks. Shortening the lifetime of an EIS to such an extent, allows for the study of the insulation materials behavior as well as investigate ways to estimate the remaining useful life (RUL) for the purpose of predictive maintenance. Unexpected failures in operation processes, where redundancy is not present, can lead to high economical losses, machine downtime and often health and safety risks. Conditions, under which thermal aging methods are generally reported in the literature, typically neglect other factors, owing to the sheer complexity and interdependence of the multifaceted aging phenomena. This paper examines some existing thermal aging tests, which are currently used to obtain data for enamel degradation in order to try to better understand of how the thermal stresses degrade the EIS. Separation of these stresses, which the EIS operate under, can yield a better understanding of how each of the Thermal, the Electrical, the Ambient and the Mechanical (TEAM) stresses behave.
       


### [Real-Time Anomaly Detection in Data Centers for Log-based Predictive Maintenance using an Evolving Fuzzy-Rule-Based Approach](https://arxiv.org/abs/2004.13527)

**Authors:**
Leticia Decker, Daniel Leite, Luca Giommi, Daniele Bonacorsi

**Abstract:**
Detection of anomalous behaviors in data centers is crucial to predictive maintenance and data safety. With data centers, we mean any computer network that allows users to transmit and exchange data and information. In particular, we focus on the Tier-1 data center of the Italian Institute for Nuclear Physics (INFN), which supports the high-energy physics experiments at the Large Hadron Collider (LHC) in Geneva. The center provides resources and services needed for data processing, storage, analysis, and distribution. Log records in the data center is a stochastic and non-stationary phenomenon in nature. We propose a real-time approach to monitor and classify log records based on sliding time windows, and a time-varying evolving fuzzy-rule-based classification model. The most frequent log pattern according to a control chart is taken as the normal system status. We extract attributes from time windows to gradually develop and update an evolving Gaussian Fuzzy Classifier (eGFC) on the fly. The real-time anomaly monitoring system has to provide encouraging results in terms of accuracy, compactness, and real-time operation.
       


### [Neural Network and Particle Filtering: A Hybrid Framework for Crack Propagation Prediction](https://arxiv.org/abs/2004.13556)

**Authors:**
Seyed Fouad Karimian, Ramin Moradi, Sergio Cofre-Martel, Katrina M. Groth, Mohammad Modarres

**Abstract:**
Crack detection, length estimation, and Remaining Useful Life (RUL) prediction are among the most studied topics in reliability engineering. Several research efforts have studied physics of failure (PoF) of different materials, along with data-driven approaches as an alternative to the traditional PoF studies. To bridge the gap between these two techniques, we propose a novel hybrid framework for fatigue crack length estimation and prediction. Physics-based modeling is performed on the fracture mechanics degradation data by estimating parameters of the Paris Law, including the associated uncertainties. Crack length estimations are inferred by feeding manually extracted features from ultrasonic signals to a Neural Network (NN). The crack length prediction is then performed using the Particle Filter (PF) approach, which takes the Paris Law as a move function and uses the NN's output as observation to update the crack growth path. This hybrid framework combines machine learning, physics-based modeling, and Bayesian updating with promising results.
       


### [Detecting Production Phases Based on Sensor Values using 1D-CNNs](https://arxiv.org/abs/2004.14475)

**Authors:**
Burkhard Hoppenstedt, Manfred Reichert, Ghada El-Khawaga, Klaus Kammerer, Karl-Michael Winter, Rüdiger Pryss

**Abstract:**
In the context of Industry 4.0, the knowledge extraction from sensor information plays an important role. Often, information gathered from sensor values reveals meaningful insights for production levels, such as anomalies or machine states. In our use case, we identify production phases through the inspection of sensor values with the help of convolutional neural networks. The data set stems from a tempering furnace used for metal heat treating. Our supervised learning approach unveils a promising accuracy for the chosen neural network that was used for the detection of production phases. We consider solutions like shown in this work as salient pillars in the field of predictive maintenance.
       


## May
### [Comparison of Evolving Granular Classifiers applied to Anomaly Detection for Predictive Maintenance in Computing Centers](https://arxiv.org/abs/2005.04156)

**Authors:**
Leticia Decker, Daniel Leite, Fabio Viola, Daniele Bonacorsi

**Abstract:**
Log-based predictive maintenance of computing centers is a main concern regarding the worldwide computing grid that supports the CERN (European Organization for Nuclear Research) physics experiments. A log, as event-oriented adhoc information, is quite often given as unstructured big data. Log data processing is a time-consuming computational task. The goal is to grab essential information from a continuously changeable grid environment to construct a classification model. Evolving granular classifiers are suited to learn from time-varying log streams and, therefore, perform online classification of the severity of anomalies. We formulated a 4-class online anomaly classification problem, and employed time windows between landmarks and two granular computing methods, namely, Fuzzy-set-Based evolving Modeling (FBeM) and evolving Granular Neural Network (eGNN), to model and monitor logging activity rate. The results of classification are of utmost importance for predictive maintenance because priority can be given to specific time intervals in which the classifier indicates the existence of high or medium severity anomalies.
       


### [System-Level Predictive Maintenance: Review of Research Literature and Gap Analysis](https://arxiv.org/abs/2005.05239)

**Authors:**
Kyle Miller, Artur Dubrawski

**Abstract:**
This paper reviews current literature in the field of predictive maintenance from the system point of view. We differentiate the existing capabilities of condition estimation and failure risk forecasting as currently applied to simple components, from the capabilities needed to solve the same tasks for complex assets. System-level analysis faces more complex latent degradation states, it has to comprehensively account for active maintenance programs at each component level and consider coupling between different maintenance actions, while reflecting increased monetary and safety costs for system failures. As a result, methods that are effective for forecasting risk and informing maintenance decisions regarding individual components do not readily scale to provide reliable sub-system or system level insights. A novel holistic modeling approach is needed to incorporate available structural and physical knowledge and naturally handle the complexities of actively fielded and maintained assets.
       


### [Synthetic Image Augmentation for Damage Region Segmentation using Conditional GAN with Structure Edge](https://arxiv.org/abs/2005.08628)

**Authors:**
Takato Yasuno, Michihiro Nakajima, Tomoharu Sekiguchi, Kazuhiro Noda, Kiyoshi Aoyanagi, Sakura Kato

**Abstract:**
Recently, social infrastructure is aging, and its predictive maintenance has become important issue. To monitor the state of infrastructures, bridge inspection is performed by human eye or bay drone. For diagnosis, primary damage region are recognized for repair targets. But, the degradation at worse level has rarely occurred, and the damage regions of interest are often narrow, so their ratio per image is extremely small pixel count, as experienced 0.6 to 1.5 percent. The both scarcity and imbalance property on the damage region of interest influences limited performance to detect damage. If additional data set of damaged images can be generated, it may enable to improve accuracy in damage region segmentation algorithm. We propose a synthetic augmentation procedure to generate damaged images using the image-to-image translation mapping from the tri-categorical label that consists the both semantic label and structure edge to the real damage image. We use the Sobel gradient operator to enhance structure edge. Actually, in case of bridge inspection, we apply the RC concrete structure with the number of 208 eye-inspection photos that rebar exposure have occurred, which are prepared 840 block images with size 224 by 224. We applied popular per-pixel segmentation algorithms such as the FCN-8s, SegNet, and DeepLabv3+Xception-v2. We demonstrates that re-training a data set added with synthetic augmentation procedure make higher accuracy based on indices the mean IoU, damage region of interest IoU, precision, recall, BF score when we predict test images.
       


### [MaintNet: A Collaborative Open-Source Library for Predictive Maintenance Language Resources](https://arxiv.org/abs/2005.12443)

**Authors:**
Farhad Akhbardeh, Travis Desell, Marcos Zampieri

**Abstract:**
Maintenance record logbooks are an emerging text type in NLP. They typically consist of free text documents with many domain specific technical terms, abbreviations, as well as non-standard spelling and grammar, which poses difficulties to NLP pipelines trained on standard corpora. Analyzing and annotating such documents is of particular importance in the development of predictive maintenance systems, which aim to provide operational efficiencies, prevent accidents and save lives. In order to facilitate and encourage research in this area, we have developed MaintNet, a collaborative open-source library of technical and domain-specific language datasets. MaintNet provides novel logbook data from the aviation, automotive, and facilities domains along with tools to aid in their (pre-)processing and clustering. Furthermore, it provides a way to encourage discussion on and sharing of new datasets and tools for logbook data analysis.
       


## June
### [Health Indicator Forecasting for Improving Remaining Useful Life Estimation](https://arxiv.org/abs/2006.03729)

**Authors:**
Qiyao Wang, Ahmed Farahat, Chetan Gupta, Haiyan Wang

**Abstract:**
Prognostics is concerned with predicting the future health of the equipment and any potential failures. With the advances in the Internet of Things (IoT), data-driven approaches for prognostics that leverage the power of machine learning models are gaining popularity. One of the most important categories of data-driven approaches relies on a predefined or learned health indicator to characterize the equipment condition up to the present time and make inference on how it is likely to evolve in the future. In these approaches, health indicator forecasting that constructs the health indicator curve over the lifespan using partially observed measurements (i.e., health indicator values within an initial period) plays a key role. Existing health indicator forecasting algorithms, such as the functional Empirical Bayesian approach, the regression-based formulation, a naive scenario matching based on the nearest neighbor, have certain limitations. In this paper, we propose a new `generative + scenario matching' algorithm for health indicator forecasting. The key idea behind the proposed approach is to first non-parametrically fit the underlying health indicator curve with a continuous Gaussian Process using a sample of run-to-failure health indicator curves. The proposed approach then generates a rich set of random curves from the learned distribution, attempting to obtain all possible variations of the target health condition evolution process over the system's lifespan. The health indicator extrapolation for a piece of functioning equipment is inferred as the generated curve that has the highest matching level within the observed period. Our experimental results show the superiority of our algorithm over the other state-of-the-art methods.
       


### [Sensor Artificial Intelligence and its Application to Space Systems -- A White Paper](https://arxiv.org/abs/2006.08368)

**Authors:**
Anko Börner, Heinz-Wilhelm Hübers, Odej Kao, Florian Schmidt, Sören Becker, Joachim Denzler, Daniel Matolin, David Haber, Sergio Lucia, Wojciech Samek, Rudolph Triebel, Sascha Eichstädt, Felix Biessmann, Anna Kruspe, Peter Jung, Manon Kok, Guillermo Gallego, Ralf Berger

**Abstract:**
Information and communication technologies have accompanied our everyday life for years. A steadily increasing number of computers, cameras, mobile devices, etc. generate more and more data, but at the same time we realize that the data can only partially be analyzed with classical approaches. The research and development of methods based on artificial intelligence (AI) made enormous progress in the area of interpretability of data in recent years. With growing experience, both, the potential and limitations of these new technologies are increasingly better understood. Typically, AI approaches start with the data from which information and directions for action are derived. However, the circumstances under which such data are collected and how they change over time are rarely considered. A closer look at the sensors and their physical properties within AI approaches will lead to more robust and widely applicable algorithms. This holistic approach which considers entire signal chains from the origin to a data product, "Sensor AI", is a highly relevant topic with great potential. It will play a decisive role in autonomous driving as well as in areas of automated production, predictive maintenance or space research. The goal of this white paper is to establish "Sensor AI" as a dedicated research topic. We want to exchange knowledge on the current state-of-the-art on Sensor AI, to identify synergies among research groups and thus boost the collaboration in this key technology for science and industry.
       


### [Uncovering the Underlying Physics of Degrading System Behavior Through a Deep Neural Network Framework: The Case of Remaining Useful Life Prognosis](https://arxiv.org/abs/2006.09288)

**Authors:**
Sergio Cofre-Martel, Enrique Lopez Droguett, Mohammad Modarres

**Abstract:**
Deep learning (DL) has become an essential tool in prognosis and health management (PHM), commonly used as a regression algorithm for the prognosis of a system's behavior. One particular metric of interest is the remaining useful life (RUL) estimated using monitoring sensor data. Most of these deep learning applications treat the algorithms as black-box functions, giving little to no control of the data interpretation. This becomes an issue if the models break the governing laws of physics or other natural sciences when no constraints are imposed. The latest research efforts have focused on applying complex DL models to achieve a low prediction error rather than studying how the models interpret the behavior of the data and the system itself. In this paper, we propose an open-box approach using a deep neural network framework to explore the physics of degradation through partial differential equations (PDEs). The framework has three stages, and it aims to discover a latent variable and corresponding PDE to represent the health state of the system. Models are trained as a supervised regression and designed to output the RUL as well as a latent variable map that can be used and interpreted as the system's health indicator.
       


### [Object Files and Schemata: Factorizing Declarative and Procedural Knowledge in Dynamical Systems](https://arxiv.org/abs/2006.16225)

**Authors:**
Anirudh Goyal, Alex Lamb, Phanideep Gampa, Philippe Beaudoin, Sergey Levine, Charles Blundell, Yoshua Bengio, Michael Mozer

**Abstract:**
Modeling a structured, dynamic environment like a video game requires keeping track of the objects and their states declarative knowledge) as well as predicting how objects behave (procedural knowledge). Black-box models with a monolithic hidden state often fail to apply procedural knowledge consistently and uniformly, i.e., they lack systematicity. For example, in a video game, correct prediction of one enemy's trajectory does not ensure correct prediction of another's. We address this issue via an architecture that factorizes declarative and procedural knowledge and that imposes modularity within each form of knowledge. The architecture consists of active modules called object files that maintain the state of a single object and invoke passive external knowledge sources called schemata that prescribe state updates. To use a video game as an illustration, two enemies of the same type will share schemata but will have separate object files to encode their distinct state (e.g., health, position). We propose to use attention to determine which object files to update, the selection of schemata, and the propagation of information between object files. The resulting architecture is a drop-in replacement conforming to the same input-output interface as normal recurrent networks (e.g., LSTM, GRU) yet achieves substantially better generalization on environments that have multiple object tokens of the same type, including a challenging intuitive physics benchmark.
       


### [Graph Neural Networks for Leveraging Industrial Equipment Structure: An application to Remaining Useful Life Estimation](https://arxiv.org/abs/2006.16556)

**Authors:**
Jyoti Narwariya, Pankaj Malhotra, Vishnu TV, Lovekesh Vig, Gautam Shroff

**Abstract:**
Automated equipment health monitoring from streaming multisensor time-series data can be used to enable condition-based maintenance, avoid sudden catastrophic failures, and ensure high operational availability. We note that most complex machinery has a well-documented and readily accessible underlying structure capturing the inter-dependencies between sub-systems or modules. Deep learning models such as those based on recurrent neural networks (RNNs) or convolutional neural networks (CNNs) fail to explicitly leverage this potentially rich source of domain-knowledge into the learning procedure. In this work, we propose to capture the structure of a complex equipment in the form of a graph, and use graph neural networks (GNNs) to model multi-sensor time-series data. Using remaining useful life estimation as an application task, we evaluate the advantage of incorporating the graph structure via GNNs on the publicly available turbofan engine benchmark dataset. We observe that the proposed GNN-based RUL estimation model compares favorably to several strong baselines from literature such as those based on RNNs and CNNs. Additionally, we observe that the learned network is able to focus on the module (node) with impending failure through a simple attention mechanism, potentially paving the way for actionable diagnosis.
       


## July
### [Analysis of Lithium-ion Battery Cells Degradation Based on Different Manufacturers](https://arxiv.org/abs/2007.01937)

**Authors:**
Ahmed Gailani, Rehab Mokidm, Moaath El-Dalahmeh, Maad El-Dalahmeh, Maher Al-Greer

**Abstract:**
Lithium-ion batteries are recognised as a key technology to power electric vehicles and integrate grid-connected renewable energy resources. The economic viability of these applications is affected by the battery degradation during its lifetime. This study presents an extensive experimental degradation data for lithium-ion battery cells from three different manufactures (Sony, BYD and Samsung). The Sony and BYD cells are of LFP chemistry while the Samsung cell is of NMC. The capacity fade and resistance increase of the battery cells are quantified due to calendar and cycle aging. The charge level and the temperature are considered as the main parameters to affect calendar aging while the depth of discharge, current rate and temperature for cycle aging. It is found that the Sony and BYD cells with LFP chemistry has calendar capacity loss of nearly 5% and 8% after 30 months respectively. Moreover, the Samsung NMC cell reached 80% state of health after 3000 cycles at 35C and 75% discharge depth suggesting a better cycle life compared to the other two battery cells with the same conditions
       


### [Predictive Maintenance for Edge-Based Sensor Networks: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/2007.03313)

**Authors:**
Kevin Shen Hoong Ong, Dusit Niyato, Chau Yuen

**Abstract:**
Failure of mission-critical equipment interrupts production and results in monetary loss. The risk of unplanned equipment downtime can be minimized through Predictive Maintenance of revenue generating assets to ensure optimal performance and safe operation of equipment. However, the increased sensorization of the equipment generates a data deluge, and existing machine-learning based predictive model alone becomes inadequate for timely equipment condition predictions. In this paper, a model-free Deep Reinforcement Learning algorithm is proposed for predictive equipment maintenance from an equipment-based sensor network context. Within each equipment, a sensor device aggregates raw sensor data, and the equipment health status is analyzed for anomalous events. Unlike traditional black-box regression models, the proposed algorithm self-learns an optimal maintenance policy and provides actionable recommendation for each equipment. Our experimental results demonstrate the potential for broader range of equipment maintenance applications as an automatic learning framework.
       


### [Predictive Analytics for Water Asset Management: Machine Learning and Survival Analysis](https://arxiv.org/abs/2007.03744)

**Authors:**
Maryam Rahbaralam, David Modesto, Jaume Cardús, Amir Abdollahi, Fernando M Cucchietti

**Abstract:**
Understanding performance and prioritizing resources for the maintenance of the drinking-water pipe network throughout its life-cycle is a key part of water asset management. Renovation of this vital network is generally hindered by the difficulty or impossibility to gain physical access to the pipes. We study a statistical and machine learning framework for the prediction of water pipe failures. We employ classical and modern classifiers for a short-term prediction and survival analysis to provide a broader perspective and long-term forecast, usually needed for the economic analysis of the renovation. To enrich these models, we introduce new predictors based on water distribution domain knowledge and employ a modern oversampling technique to remedy the high imbalance coming from the few failures observed each year. For our case study, we use a dataset containing the failure records of all pipes within the water distribution network in Barcelona, Spain. The results shed light on the effect of important risk factors, such as pipe geometry, age, material, and soil cover, among others, and can help utility managers conduct more informed predictive maintenance tasks.
       


### [Attention Sequence to Sequence Model for Machine Remaining Useful Life Prediction](https://arxiv.org/abs/2007.09868)

**Authors:**
Mohamed Ragab, Zhenghua Chen, Min Wu, Chee-Keong Kwoh, Ruqiang Yan, Xiaoli Li

**Abstract:**
Accurate estimation of remaining useful life (RUL) of industrial equipment can enable advanced maintenance schedules, increase equipment availability and reduce operational costs. However, existing deep learning methods for RUL prediction are not completely successful due to the following two reasons. First, relying on a single objective function to estimate the RUL will limit the learned representations and thus affect the prediction accuracy. Second, while longer sequences are more informative for modelling the sensor dynamics of equipment, existing methods are less effective to deal with very long sequences, as they mainly focus on the latest information. To address these two problems, we develop a novel attention-based sequence to sequence with auxiliary task (ATS2S) model. In particular, our model jointly optimizes both reconstruction loss to empower our model with predictive capabilities (by predicting next input sequence given current input sequence) and RUL prediction loss to minimize the difference between the predicted RUL and actual RUL. Furthermore, to better handle longer sequence, we employ the attention mechanism to focus on all the important input information during training process. Finally, we propose a new dual-latent feature representation to integrate the encoder features and decoder hidden states, to capture rich semantic information in data. We conduct extensive experiments on four real datasets to evaluate the efficacy of the proposed method. Experimental results show that our proposed method can achieve superior performance over 13 state-of-the-art methods consistently.
       


### [Artificial neural networks for disease trajectory prediction in the context of sepsis](https://arxiv.org/abs/2007.14542)

**Authors:**
Dale Larie, Gary An, Chase Cockrell

**Abstract:**
The disease trajectory for clinical sepsis, in terms of temporal cytokine and phenotypic dynamics, can be interpreted as a random dynamical system. The ability to make accurate predictions about patient state from clinical measurements has eluded the biomedical community, primarily due to the paucity of relevant and high-resolution data. We have utilized two distinct neural network architectures, Long Short-Term Memory and Multi-Layer Perceptron, to take a time sequence of five measurements of eleven simulated serum cytokine concentrations as input and to return both the future cytokine trajectories as well as an aggregate metric representing the patient's state of health. The neural networks converged within 50 epochs for cytokine trajectory predictions and health-metric regressions, with the expected amount of error (due to stochasticity in the simulation). The mapping from a specific cytokine profile to a state-of-health is not unique, and increased levels of inflammation result in less accurate predictions. Due to the propagation of machine learning error combined with computational model stochasticity over time, the network should be re-grounded in reality daily as predictions can diverge from the true model trajectory as the system evolves towards a probabilistic basin of attraction. This work serves as a proof-of-concept for the use of artificial neural networks to predict disease progression in sepsis. This work is not intended to replace a trained clinician, rather the goal is to augment intuition with quantifiable statistical information to help them make the best decisions. We note that this relies on a valid computational model of the system in question as there does not exist sufficient data to inform a machine-learning trained, artificially intelligent, controller.
       


## August
### [Automatic Remaining Useful Life Estimation Framework with Embedded Convolutional LSTM as the Backbone](https://arxiv.org/abs/2008.03961)

**Authors:**
Yexu Zhou, Yuting Gao, Yiran Huang, Michael Hefenbrock, Till Riedel, Michael Beigl

**Abstract:**
An essential task in predictive maintenance is the prediction of the Remaining Useful Life (RUL) through the analysis of multivariate time series. Using the sliding window method, Convolutional Neural Network (CNN) and conventional Recurrent Neural Network (RNN) approaches have produced impressive results on this matter, due to their ability to learn optimized features. However, sequence information is only partially modeled by CNN approaches. Due to the flatten mechanism in conventional RNNs, like Long Short Term Memories (LSTM), the temporal information within the window is not fully preserved. To exploit the multi-level temporal information, many approaches are proposed which combine CNN and RNN models. In this work, we propose a new LSTM variant called embedded convolutional LSTM (ECLSTM). In ECLSTM a group of different 1D convolutions is embedded into the LSTM structure. Through this, the temporal information is preserved between and within windows. Since the hyper-parameters of models require careful tuning, we also propose an automated prediction framework based on the Bayesian optimization with hyperband optimizer, which allows for efficient optimization of the network architecture. Finally, we show the superiority of our proposed ECLSTM approach over the state-of-the-art approaches on several widely used benchmark data sets for RUL Estimation.
       


### [Trust-Based Cloud Machine Learning Model Selection For Industrial IoT and Smart City Services](https://arxiv.org/abs/2008.05042)

**Authors:**
Basheer Qolomany, Ihab Mohammed, Ala Al-Fuqaha, Mohsen Guizan, Junaid Qadir

**Abstract:**
With Machine Learning (ML) services now used in a number of mission-critical human-facing domains, ensuring the integrity and trustworthiness of ML models becomes all-important. In this work, we consider the paradigm where cloud service providers collect big data from resource-constrained devices for building ML-based prediction models that are then sent back to be run locally on the intermittently-connected resource-constrained devices. Our proposed solution comprises an intelligent polynomial-time heuristic that maximizes the level of trust of ML models by selecting and switching between a subset of the ML models from a superset of models in order to maximize the trustworthiness while respecting the given reconfiguration budget/rate and reducing the cloud communication overhead. We evaluate the performance of our proposed heuristic using two case studies. First, we consider Industrial IoT (IIoT) services, and as a proxy for this setting, we use the turbofan engine degradation simulation dataset to predict the remaining useful life of an engine. Our results in this setting show that the trust level of the selected models is 0.49% to 3.17% less compared to the results obtained using Integer Linear Programming (ILP). Second, we consider Smart Cities services, and as a proxy of this setting, we use an experimental transportation dataset to predict the number of cars. Our results show that the selected model's trust level is 0.7% to 2.53% less compared to the results obtained using ILP. We also show that our proposed heuristic achieves an optimal competitive ratio in a polynomial-time approximation scheme for the problem.
       


### [On-line Capacity Estimation for Lithium-ion Battery Cells via an Electrochemical Model-based Adaptive Interconnected Observer](https://arxiv.org/abs/2008.10467)

**Authors:**
Anirudh Allam, Simona Onori

**Abstract:**
Battery aging is a natural process that contributes to capacity and power fade, resulting in a gradual performance degradation over time and usage. State of Charge (SOC) and State of Health (SOH) monitoring of an aging battery poses a challenging task to the Battery Management System (BMS) due to the lack of direct measurements. Estimation algorithms based on an electrochemical model that take into account the impact of aging on physical battery parameters can provide accurate information on lithium concentration and cell capacity over a battery's usable lifespan. A temperature-dependent electrochemical model, the Enhanced Single Particle Model (ESPM), forms the basis for the synthesis of an adaptive interconnected observer that exploits the relationship between capacity and power fade, due to the growth of Solid Electrolyte Interphase layer (SEI), to enable combined estimation of states (lithium concentration in both electrodes and cell capacity) and aging-sensitive transport parameters (anode diffusion coefficient and SEI layer ionic conductivity). The practical stability conditions for the adaptive observer are derived using Lyapunov's theory. Validation results against experimental data show a bounded capacity estimation error within 2% of its true value. Further, effectiveness of capacity estimation is tested for two cells at different stages of aging. Robustness of capacity estimates under measurement noise and sensor bias are studied.
       


### [An Economic Perspective on Predictive Maintenance of Filtration Units](https://arxiv.org/abs/2008.11070)

**Authors:**
Denis Tan Jing Yu, Adrian Law Wing-Keung

**Abstract:**
This paper provides an economic perspective on the predictive maintenance of filtration units. The rise of predictive maintenance is possible due to the growing trend of industry 4.0 and the availability of inexpensive sensors. However, the adoption rate for predictive maintenance by companies remains low. The majority of companies are sticking to corrective and preventive maintenance. This is not due to a lack of information on the technical implementation of predictive maintenance, with an abundance of research papers on state-of-the-art machine learning algorithms that can be used effectively. The main issue is that most upper management has not yet been fully convinced of the idea of predictive maintenance. The economic value of the implementation has to be linked to the predictive maintenance program for better justification by the management. In this study, three machine learning models were trained to demonstrate the economic value of predictive maintenance. Data was collected from a testbed located at the Singapore University of Technology and Design. The testbed closely resembles a real-world water treatment plant. A cost-benefit analysis coupled with Monte Carlo simulation was proposed. It provided a structured approach to document potential costs and savings by implementing a predictive maintenance program. The simulation incorporated real-world risk into a financial model. Financial figures were adapted from CITIC Envirotech Ltd, a leading membrane-based integrated environmental solutions provider. Two scenarios were used to elaborate on the economic values of predictive maintenance. Overall, this study seeks to bridge the gap between technical and business domains of predictive maintenance.
       


## September
### [Advancing from Predictive Maintenance to Intelligent Maintenance with AI and IIoT](https://arxiv.org/abs/2009.00351)

**Authors:**
Haining Zheng, Antonio R. Paiva, Chris S. Gurciullo

**Abstract:**
As Artificial Intelligent (AI) technology advances and increasingly large amounts of data become readily available via various Industrial Internet of Things (IIoT) projects, we evaluate the state of the art of predictive maintenance approaches and propose our innovative framework to improve the current practice. The paper first reviews the evolution of reliability modelling technology in the past 90 years and discusses major technologies developed in industry and academia. We then introduce the next generation maintenance framework - Intelligent Maintenance, and discuss its key components. This AI and IIoT based Intelligent Maintenance framework is composed of (1) latest machine learning algorithms including probabilistic reliability modelling with deep learning, (2) real-time data collection, transfer, and storage through wireless smart sensors, (3) Big Data technologies, (4) continuously integration and deployment of machine learning models, (5) mobile device and AR/VR applications for fast and better decision-making in the field. Particularly, we proposed a novel probabilistic deep learning reliability modelling approach and demonstrate it in the Turbofan Engine Degradation Dataset.
       


### [Crafting Adversarial Examples for Deep Learning Based Prognostics (Extended Version)](https://arxiv.org/abs/2009.10149)

**Authors:**
Gautam Raj Mode, Khaza Anuarul Hoque

**Abstract:**
In manufacturing, unexpected failures are considered a primary operational risk, as they can hinder productivity and can incur huge losses. State-of-the-art Prognostics and Health Management (PHM) systems incorporate Deep Learning (DL) algorithms and Internet of Things (IoT) devices to ascertain the health status of equipment, and thus reduce the downtime, maintenance cost and increase the productivity. Unfortunately, IoT sensors and DL algorithms, both are vulnerable to cyber attacks, and hence pose a significant threat to PHM systems. In this paper, we adopt the adversarial example crafting techniques from the computer vision domain and apply them to the PHM domain. Specifically, we craft adversarial examples using the Fast Gradient Sign Method (FGSM) and Basic Iterative Method (BIM) and apply them on the Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Convolutional Neural Network (CNN) based PHM models. We evaluate the impact of adversarial attacks using NASA's turbofan engine dataset. The obtained results show that all the evaluated PHM models are vulnerable to adversarial attacks and can cause a serious defect in the remaining useful life estimation. The obtained results also show that the crafted adversarial examples are highly transferable and may cause significant damages to PHM systems.
       


### [Goals and Measures for Analyzing Power Consumption Data in Manufacturing Enterprises](https://arxiv.org/abs/2009.10369)

**Authors:**
Sören Henning, Wilhelm Hasselbring, Heinz Burmester, Armin Möbius, Maik Wojcieszak

**Abstract:**
The Internet of Things adoption in the manufacturing industry allows enterprises to monitor their electrical power consumption in real time and at machine level. In this paper, we follow up on such emerging opportunities for data acquisition and show that analyzing power consumption in manufacturing enterprises can serve a variety of purposes. Apart from the prevalent goal of reducing overall power consumption for economical and ecological reasons, such data can, for example, be used to improve production processes.
  Based on a literature review and expert interviews, we discuss how analyzing power consumption data can serve the goals reporting, optimization, fault detection, and predictive maintenance. To tackle these goals, we propose to implement the measures real-time data processing, multi-level monitoring, temporal aggregation, correlation, anomaly detection, forecasting, visualization, and alerting in software.
  We transfer our findings to two manufacturing enterprises and show how the presented goals reflect in these enterprises. In a pilot implementation of a power consumption analytics platform, we show how our proposed measures can be implemented with a microservice-based architecture, stream processing techniques, and the fog computing paradigm. We provide the implementations as open source as well as a public demo allowing to reproduce and extend our research.
       


### [Using Machine Learning to Develop a Novel COVID-19 Vulnerability Index (C19VI)](https://arxiv.org/abs/2009.10808)

**Authors:**
Anuj Tiwari, Arya V. Dadhania, Vijay Avin Balaji Ragunathrao, Edson R. A. Oliveira

**Abstract:**
COVID19 is now one of the most leading causes of death in the United States. Systemic health, social and economic disparities have put the minorities and economically poor communities at a higher risk than others. There is an immediate requirement to develop a reliable measure of county-level vulnerabilities that can capture the heterogeneity of both vulnerable communities and the COVID19 pandemic. This study reports a COVID19 Vulnerability Index (C19VI) for identification and mapping of vulnerable counties in the United States. We proposed a Random Forest machine learning based COVID19 vulnerability model using CDC sociodemographic and COVID19-specific themes. An innovative COVID19 Impact Assessment algorithm was also developed using homogeneity and trend assessment technique for evaluating severity of the pandemic in all counties and train RF model. Developed C19VI was statistically validated and compared with the CDC COVID19 Community Vulnerability Index (CCVI). Finally, using C19VI along with census data, we explored racial inequalities and economic disparities in COVID19 health outcomes amongst different regions in the United States. Our C19VI index indicates that 18.30% of the counties falls into very high vulnerability class, 24.34% in high, 23.32% in moderate, 22.34% in low, and 11.68% in very low. Furthermore, C19VI reveals that 75.57% of racial minorities and 82.84% of economically poor communities are very high or high COVID19 vulnerable regions. The proposed approach of vulnerability modeling takes advantage of both the well-established field of statistical analysis and the fast-evolving domain of machine learning. C19VI provides an accurate and more reliable way to measure county level vulnerability in the United States. This index aims at helping emergency planners to develop more effective mitigation strategies especially for the disproportionately impacted communities.
       


### [Computational framework for real-time diagnostics and prognostics of aircraft actuation systems](https://arxiv.org/abs/2009.14645)

**Authors:**
Pier Carlo Berri, Matteo D. L. Dalla Vedova, Laura Mainini

**Abstract:**
Prognostics and Health Management (PHM) are emerging approaches to product life cycle that will maintain system safety and improve reliability, while reducing operating and maintenance costs. This is particularly relevant for aerospace systems, where high levels of integrity and high performances are required at the same time. We propose a novel strategy for the nearly real-time Fault Detection and Identification (FDI) of a dynamical assembly, and for the estimation of Remaining Useful Life (RUL) of the system. The availability of a timely estimate of the health status of the system will allow for an informed adaptive planning of maintenance and a dynamical reconfiguration of the mission profile, reducing operating costs and improving reliability. This work addresses the three phases of the prognostic flow - namely (1) signal acquisition, (2) Fault Detection and Identification, and (3) Remaining Useful Life estimation - and introduces a computationally efficient procedure suitable for real-time, on-board execution. To achieve this goal, we propose to combine information from physical models of different fidelity with machine learning techniques to obtain efficient representations (surrogate models) suitable for nearly real-time applications. Additionally, we propose an importance sampling strategy and a novel approach to model damage propagation for dynamical systems. The methodology is assessed for the FDI and RUL estimation of an aircraft electromechanical actuator (EMA) for secondary flight controls. The results show that the proposed method allows for a high precision in the evaluation of the system RUL, while outperforming common model-based techniques in terms of computational time.
       


## October
### [Deep learning models for predictive maintenance: a survey, comparison, challenges and prospect](https://arxiv.org/abs/2010.03207)

**Authors:**
Oscar Serradilla, Ekhi Zugasti, Urko Zurutuza

**Abstract:**
Given the growing amount of industrial data spaces worldwide, deep learning solutions have become popular for predictive maintenance, which monitor assets to optimise maintenance tasks. Choosing the most suitable architecture for each use-case is complex given the number of examples found in literature. This work aims at facilitating this task by reviewing state-of-the-art deep learning architectures, and how they integrate with predictive maintenance stages to meet industrial companies' requirements (i.e. anomaly detection, root cause analysis, remaining useful life estimation). They are categorised and compared in industrial applications, explaining how to fill their gaps. Finally, open challenges and future research paths are presented.
       


### [A Reinforcement Learning Approach to Health Aware Control Strategy](https://arxiv.org/abs/2010.09269)

**Authors:**
Mayank Shekhar Jha, Philippe Weber, Didier Theilliol, Jean-Christophe Ponsart, Didier Maquin

**Abstract:**
Health-aware control (HAC) has emerged as one of the domains where control synthesis is sought based upon the failure prognostics of system/component or the Remaining Useful Life (RUL) predictions of critical components. The fact that mathematical dynamic (transition) models of RUL are rarely available, makes it difficult for RUL information to be incorporated into the control paradigm. A novel framework for health aware control is presented in this paper where reinforcement learning based approach is used to learn an optimal control policy in face of component degradation by integrating global system transition data (generated by an analytical model that mimics the real system) and RUL predictions. The RUL predictions generated at each step, is tracked to a desired value of RUL. The latter is integrated within a cost function which is maximized to learn the optimal control. The proposed method is studied using simulation of a DC motor and shaft wear.
       


### [Robust State of Health Estimation of Lithium-ion Batteries Using Convolutional Neural Network and Random Forest](https://arxiv.org/abs/2010.10452)

**Authors:**
Niankai Yang, Ziyou Song, Heath Hofmann, Jing Sun

**Abstract:**
The State of Health (SOH) of lithium-ion batteries is directly related to their safety and efficiency, yet effective assessment of SOH remains challenging for real-world applications (e.g., electric vehicle). In this paper, the estimation of SOH (i.e., capacity fading) under partial discharge with different starting and final State of Charge (SOC) levels is investigated. The challenge lies in the fact that partial discharge truncates the data available for SOH estimation, thereby leading to the loss or distortion of common SOH indicators. To address this challenge associated with partial discharge, we explore the convolutional neural network (CNN) to extract indicators for both SOH and changes in SOH ($Δ$SOH) between two successive charge/discharge cycles. The random forest algorithm is then adopted to produce the final SOH estimate by exploiting the indicators from the CNNs. Performance evaluation is conducted using the partial discharge data with different SOC ranges created from a fast-discharging dataset. The proposed approach is compared with i) a differential analysis-based approach and ii) two CNN-based approaches using only SOH and $Δ$SOH indicators, respectively. Through comparison, the proposed approach demonstrates improved estimation accuracy and robustness. Sensitivity analysis of the CNN and random forest models further validates that the proposed approach makes better use of the available partial discharge data for SOH estimation.
       


### [Smart Anomaly Detection in Sensor Systems: A Multi-Perspective Review](https://arxiv.org/abs/2010.14946)

**Authors:**
L. Erhan, M. Ndubuaku, M. Di Mauro, W. Song, M. Chen, G. Fortino, O. Bagdasar, A. Liotta

**Abstract:**
Anomaly detection is concerned with identifying data patterns that deviate remarkably from the expected behaviour. This is an important research problem, due to its broad set of application domains, from data analysis to e-health, cybersecurity, predictive maintenance, fault prevention, and industrial automation. Herein, we review state-of-the-art methods that may be employed to detect anomalies in the specific area of sensor systems, which poses hard challenges in terms of information fusion, data volumes, data speed, and network/energy efficiency, to mention but the most pressing ones. In this context, anomaly detection is a particularly hard problem, given the need to find computing-energy accuracy trade-offs in a constrained environment. We taxonomize methods ranging from conventional techniques (statistical methods, time-series analysis, signal processing, etc.) to data-driven techniques (supervised learning, reinforcement learning, deep learning, etc.). We also look at the impact that different architectural environments (Cloud, Fog, Edge) can have on the sensors ecosystem. The review points to the most promising intelligent-sensing methods, and pinpoints a set of interesting open issues and challenges.
       


### [Automatic joint damage quantification using computer vision and deep learning](https://arxiv.org/abs/2010.15303)

**Authors:**
Quang Tran, Jeffery R. Roesler

**Abstract:**
Joint raveled or spalled damage (henceforth called joint damage) can affect the safety and long-term performance of concrete pavements. It is important to assess and quantify the joint damage over time to assist in building action plans for maintenance, predicting maintenance costs, and maximize the concrete pavement service life. A framework for the accurate, autonomous, and rapid quantification of joint damage with a low-cost camera is proposed using a computer vision technique with a deep learning (DL) algorithm. The DL model is employed to train 263 images of sawcuts with joint damage. The trained DL model is used for pixel-wise color-masking joint damage in a series of query 2D images, which are used to reconstruct a 3D image using open-source structure from motion algorithm. Another damage quantification algorithm using a color threshold is applied to detect and compute the surface area of the damage in the 3D reconstructed image. The effectiveness of the framework was validated through inspecting joint damage at four transverse contraction joints in Illinois, USA, including three acceptable joints and one unacceptable joint by visual inspection. The results show the framework achieves 76% recall and 10% error.
       


## November
### [Discovering long term dependencies in noisy time series data using deep learning](https://arxiv.org/abs/2011.07551)

**Author:**
Alexey Kurochkin

**Abstract:**
Time series modelling is essential for solving tasks such as predictive maintenance, quality control and optimisation. Deep learning is widely used for solving such problems. When managing complex manufacturing process with neural networks, engineers need to know why machine learning model made specific decision and what are possible outcomes of following model recommendation. In this paper we develop framework for capturing and explaining temporal dependencies in time series data using deep neural networks and test it on various synthetic and real world datasets.
       


### [Dynamic Valuation of Battery Lifetime](https://arxiv.org/abs/2011.08425)

**Author:**
Bolun Xu

**Abstract:**
This paper proposes a dynamic valuation framework to determine the opportunity value of battery capacity degradation in grid applications based on the internal degradation mechanism and utilization scenarios. The proposed framework follows a dynamic programming approach and includes a piecewise linear value function approximation solution that solves the optimization problem over a long planning horizon. The paper provides two case studies on price arbitrage and frequency regulation using real market and system data to demonstrate the broad applicability of the proposed framework. Results show that the battery lifetime value is critically dependent on both the external market environment and its internal state of health. On the grid service side, results show that second-life batteries can provide more than 50% of the value compared to new batteries, and frequency regulation provides two times more revenue than price arbitrage throughout the battery lifetime.
       


### [Tracking and Visualizing Signs of Degradation for an Early Failure Prediction of a Rolling Bearing](https://arxiv.org/abs/2011.09086)

**Authors:**
Sana Talmoudi, Tetsuya Kanada, Yasuhisa Hirata

**Abstract:**
Predictive maintenance, i.e. predicting failure to be few steps ahead of the fault, is one of the pillars of Industry 4.0. An effective method for that is to track early signs of degradation before a failure happens. This paper presents an innovative failure predictive scheme for machines. The proposed scheme combines the use of full spectrum of the vibration data caused by the machines and data visualization technologies. This scheme is featured by no training data required and by quick start after installation. First, we propose to use full spectrum (as high-dimensional data vector) with no cropping and no complex feature extraction and to visualize data behavior by mapping the high dimensional vectors into a 2D map. We then can ensure the simplicity of process and less possibility of overlooking of important information as well as providing a human-friendly and human-understandable output. Second, we propose Real-Time Data Tracker (RTDT) which predicts the failure at an appropriate time with sufficient time for maintenance by plotting real-time frequency spectrum data of the target machine on the 2D map composed from normal data. Third, we show the test results of our proposal using vibration data of bearings from real-world test-to-failure measurements provided by the public dataset, the IMS dataset.
       


### [Predictive maintenance on event logs: Application on an ATM fleet](https://arxiv.org/abs/2011.10996)

**Authors:**
Antoine Guillaume, Christel Vrain, Elloumi Wael

**Abstract:**
Predictive maintenance is used in industrial applications to increase machine availability and optimize cost related to unplanned maintenance. In most cases, predictive maintenance applications use output from sensors, recording physical phenomenons such as temperature or vibration which can be directly linked to the degradation process of the machine. However, in some applications, outputs from sensors are not available, and event logs generated by the machine are used instead. We first study the approaches used in the literature to solve predictive maintenance problems and present a new public dataset containing the event logs from 156 machines. After this, we define an evaluation framework for predictive maintenance systems, which takes into account business constraints, and conduct experiments to explore suitable solutions, which can serve as guidelines for future works using this new dataset.
       


### [Smart cathodic protection system for real-time quantitative assessment of corrosion of sacrificial anode based on Electro-Mechanical Impedance (EMI)](https://arxiv.org/abs/2011.11011)

**Authors:**
Durgesh Tamhane, Jeslin Thalapil, Sauvik Banerjee, Siddharth Tallur

**Abstract:**
Corrosion of metal structures is often prevented using cathodic protection systems, that employ sacrificial anodes that corrode more preferentially relative to the metal to be protected. In-situ monitoring of these sacrificial anodes during early stages of their useful life could offer several insights into deterioration of the material surrounding the infrastructure as well as serve as early warning indicator for preventive maintenance of critical infrastructure. In this paper, we present an Electro-Mechanical Impedance (EMI) measurement-based technique to quantify extent of corrosion of a zinc sacrificial anode without manual intervention. The detection apparatus consists of a lead zirconate titanate (PZT) transducer affixed onto a circular zinc disc, with waterproofing epoxy protecting the transducer element when the assembly is submerged in liquid electrolyte (salt solution) for accelerated corrosion by means of impressed current. We develop an analytical model for discerning the extent of corrosion by monitoring shift in resonance frequency for in-plane radial expansion mode of the disc, that also accurately models the nonlinearity introduced by partial delamination of the corrosion product (zinc oxide) from the disc. The analytical model thus developed shows excellent agreement with Finite Element Analysis (FEA) and experimental results. Our work establishes the efficacy of the proposed technique for monitoring the state of health of sacrificial anodes in their early stage of deterioration and could thus be widely adopted for structural health monitoring applications within the internet of things.
       


### [Remaining Useful Life Estimation Under Uncertainty with Causal GraphNets](https://arxiv.org/abs/2011.11740)

**Authors:**
Charilaos Mylonas, Eleni Chatzi

**Abstract:**
In this work, a novel approach for the construction and training of time series models is presented that deals with the problem of learning on large time series with non-equispaced observations, which at the same time may possess features of interest that span multiple scales. The proposed method is appropriate for constructing predictive models for non-stationary stochastic time series.The efficacy of the method is demonstrated on a simulated stochastic degradation dataset and on a real-world accelerated life testing dataset for ball-bearings. The proposed method, which is based on GraphNets, implicitly learns a model that describes the evolution of the system at the level of a state-vector rather than of a raw observation. The proposed approach is compared to a recurrent network with a temporal convolutional feature extractor head (RNN-tCNN) which forms a known viable alternative for the problem context considered. Finally, by taking advantage of recent advances in the computation of reparametrization gradients for learning probability distributions, a simple yet effective technique for representing prediction uncertainty as a Gamma distribution over remaining useful life predictions is employed.
       


### [A Framework for Health-informed RUL-constrained Optimal Power Flow with Li-ion Batteries](https://arxiv.org/abs/2011.14318)

**Authors:**
Jiahang Xie, Yu Weng, Hung D. Nguyen

**Abstract:**
Battery energy storage systems are widely adopted in grid-connected applications to mitigate the impact of intermittent renewable generations and enhance power system resiliency. Degradation of the battery during its service time is one of the major concerns in the deployment that strongly affects the long-term lifetime. Apart from environmental factors, this intrinsic property of a battery depends on the daily operating conditions. Thus, optimally engaging the daily operation of the battery based on its current status in order to meet the required remaining useful life becomes a practical and demanding need. To address this issue, this paper proposes a health-informed RUL-constrained optimal power flow framework to characterize the corresponding optimal feasible operation space. The targeted service lifespan is achieved if the battery's working condition is confined within this feasible domain. Equivalent box constraints are then constructed for better computational efficiency in solving the optimization problem. In this framework, a Monte Carlo-based data-driven approach and a health indicator (HI) representing the battery's current states are introduced. The performance of the proposed method is illustrated with the IEEE 39-bus system.
       


## December
### [Stochastic processes and host-parasite coevolution: linking coevolutionary dynamics and DNA polymorphism data](https://arxiv.org/abs/2012.02831)

**Authors:**
Wolfgang Stephan, Aurélien Tellier

**Abstract:**
Between-species coevolution, and in particular antagonistic host-parasite coevolution, is a major process shaping within-species diversity. In this paper we investigate the role of various stochastic processes affecting the outcome of the deterministic coevolutionary models. Specifically, we assess 1) the impact of genetic drift and mutation on the maintenance of polymorphism at the interacting loci, and 2) the change in neutral allele frequencies across the genome of both coevolving species due to co-demographic population size changes. We find that genetic drift decreases the likelihood to observe classic balancing selection signatures, and that for most realistic values of the coevolutionary parameters, balancing selection signatures cannot be seen at the host loci. Further, we reveal that contrary to classic expectations, fast changes in parasite population size due to eco-evo feedbacks can be tracked by the allelic site-frequency spectrum measured at several time points. Changes in host population size are, however, less pronounced and thus not observable. Finally, we also review several understudied stochastic processes occurring in host-parasite coevolution which are of importance to predict maintenance of polymorphism at the underlying loci and the genome-wide nucleotide diversity of host and parasite populations.
       


### [Fuzzy model identification based on mixture distribution analysis for bearings remaining useful life estimation using small training data set](https://arxiv.org/abs/2012.04589)

**Authors:**
Fei Huang, Alexandre Sava, Kondo H. Adjallah, Wang Zhouhang

**Abstract:**
The research work presented in this paper proposes a data-driven modeling method for bearings remaining useful life estimation based on Takagi-Sugeno (T-S) fuzzy inference system (FIS). This method allows identifying the parameters of a classic T-S FIS, starting with a small quantity of data. In this work, we used the vibration signals data from a small number of bearings over an entire period of run-to-failure. The FIS model inputs are features extracted from the vibration signals data observed periodically on the training bearings. The number of rules and the input parameters of each rule of the FIS model are identified using the subtractive clustering method. Furthermore, we propose to use the maximum likelihood method of mixture distribution analysis to calculate the parameters of clusters on the time axis and the probability corresponding to rules on degradation stages. Based on this result, we identified the output parameters of each rule using a weighted least square estimation. We then benchmarked the proposed method with some existing methods from the literature, through numerical experiments conducted on available datasets to highlight its effectiveness.
       


# 2021
## January
### [Joint Prediction of Remaining Useful Life and Failure Type of Train Wheelsets: A Multi-task Learning Approach](https://arxiv.org/abs/2101.03497)

**Author:**
Weixin Wang

**Abstract:**
The failures of train wheels account for disruptions of train operations and even a large portion of train derailments. Remaining useful life (RUL) of a wheelset measures the how soon the next failure will arrive, and the failure type reveals how severe the failure will be. RUL prediction is a regression task, whereas failure type is a classification task. In this paper, we propose a multi-task learning approach to jointly accomplish these two tasks by using a common input space to achieve more desirable results. We develop a convex optimization formulation to integrate both least square loss and the negative maximum likelihood of logistic regression, and model the joint sparsity as the L2/L1 norm of the model parameters to couple feature selection across tasks. The experiment results show that our method outperforms the single task learning method by 3% in prediction accuracy.
       


### [Time-Series Regeneration with Convolutional Recurrent Generative Adversarial Network for Remaining Useful Life Estimation](https://arxiv.org/abs/2101.03678)

**Authors:**
Xuewen Zhang, Yan Qin, Chau Yuen, Lahiru Jayasinghe, Xiang Liu

**Abstract:**
For health prognostic task, ever-increasing efforts have been focused on machine learning-based methods, which are capable of yielding accurate remaining useful life (RUL) estimation for industrial equipment or components without exploring the degradation mechanism. A prerequisite ensuring the success of these methods depends on a wealth of run-to-failure data, however, run-to-failure data may be insufficient in practice. That is, conducting a substantial amount of destructive experiments not only is high costs, but also may cause catastrophic consequences. Out of this consideration, an enhanced RUL framework focusing on data self-generation is put forward for both non-cyclic and cyclic degradation patterns for the first time. It is designed to enrich data from a data-driven way, generating realistic-like time-series to enhance current RUL methods. First, high-quality data generation is ensured through the proposed convolutional recurrent generative adversarial network (CR-GAN), which adopts a two-channel fusion convolutional recurrent neural network. Next, a hierarchical framework is proposed to combine generated data into current RUL estimation methods. Finally, the efficacy of the proposed method is verified through both non-cyclic and cyclic degradation systems. With the enhanced RUL framework, an aero-engine system following non-cyclic degradation has been tested using three typical RUL models. State-of-art RUL estimation results are achieved by enhancing capsule network with generated time-series. Specifically, estimation errors evaluated by the index score function have been reduced by 21.77%, and 32.67% for the two employed operating conditions, respectively. Besides, the estimation error is reduced to zero for the Lithium-ion battery system, which presents cyclic degradation.
       


### [Analysis of key flavors of event-driven predictive maintenance using logs of phenomena described by Weibull distributions](https://arxiv.org/abs/2101.07033)

**Authors:**
Petros Petsinis, Athanasios Naskos, Anastasios Gounaris

**Abstract:**
This work explores two approaches to event-driven predictive maintenance in Industry 4.0 that cast the problem at hand as a classification or a regression one, respectively, using as a starting point two state-of-the-art solutions. For each of the two approaches, we examine different data preprocessing techniques, different prediction algorithms and the impact of ensemble and sampling methods. Through systematic experiments regarding the aspectsmentioned above,we aimto understand the strengths of the alternatives, and more importantly, shed light on how to navigate through the vast number of such alternatives in an informed manner. Our work constitutes a key step towards understanding the true potential of this type of data-driven predictive maintenance as of to date, and assist practitioners in focusing on the aspects that have the greatest impact.
       


## February
### [Machine learning pipeline for battery state of health estimation](https://arxiv.org/abs/2102.00837)

**Authors:**
Darius Roman, Saurabh Saxena, Valentin Robu, Michael Pecht, David Flynn

**Abstract:**
Lithium-ion batteries are ubiquitous in modern day applications ranging from portable electronics to electric vehicles. Irrespective of the application, reliable real-time estimation of battery state of health (SOH) by on-board computers is crucial to the safe operation of the battery, ultimately safeguarding asset integrity. In this paper, we design and evaluate a machine learning pipeline for estimation of battery capacity fade - a metric of battery health - on 179 cells cycled under various conditions. The pipeline estimates battery SOH with an associated confidence interval by using two parametric and two non-parametric algorithms. Using segments of charge voltage and current curves, the pipeline engineers 30 features, performs automatic feature selection and calibrates the algorithms. When deployed on cells operated under the fast-charging protocol, the best model achieves a root mean squared percent error of 0.45\%. This work provides insights into the design of scalable data-driven models for battery SOH estimation, emphasising the value of confidence bounds around the prediction. The pipeline methodology combines experimental data with machine learning modelling and can be generalized to other critical components that require real-time estimation of SOH.
       


### [AttDMM: An Attentive Deep Markov Model for Risk Scoring in Intensive Care Units](https://arxiv.org/abs/2102.04702)

**Authors:**
Yilmazcan Özyurt, Mathias Kraus, Tobias Hatt, Stefan Feuerriegel

**Abstract:**
Clinical practice in intensive care units (ICUs) requires early warnings when a patient's condition is about to deteriorate so that preventive measures can be undertaken. To this end, prediction algorithms have been developed that estimate the risk of mortality in ICUs. In this work, we propose a novel generative deep probabilistic model for real-time risk scoring in ICUs. Specifically, we develop an attentive deep Markov model called AttDMM. To the best of our knowledge, AttDMM is the first ICU prediction model that jointly learns both long-term disease dynamics (via attention) and different disease states in health trajectory (via a latent variable model). Our evaluations were based on an established baseline dataset (MIMIC-III) with 53,423 ICU stays. The results confirm that compared to state-of-the-art baselines, our AttDMM was superior: AttDMM achieved an area under the receiver operating characteristic curve (AUROC) of 0.876, which yielded an improvement over the state-of-the-art method by 2.2%. In addition, the risk score from the AttDMM provided warnings several hours earlier. Thereby, our model shows a path towards identifying patients at risk so that health practitioners can intervene early and save patient lives.
       


### [Anomaly Detection through Transfer Learning in Agriculture and Manufacturing IoT Systems](https://arxiv.org/abs/2102.05814)

**Authors:**
Mustafa Abdallah, Wo Jae Lee, Nithin Raghunathan, Charilaos Mousoulis, John W. Sutherland, Saurabh Bagchi

**Abstract:**
IoT systems have been facing increasingly sophisticated technical problems due to the growing complexity of these systems and their fast deployment practices. Consequently, IoT managers have to judiciously detect failures (anomalies) in order to reduce their cyber risk and operational cost. While there is a rich literature on anomaly detection in many IoT-based systems, there is no existing work that documents the use of ML models for anomaly detection in digital agriculture and in smart manufacturing systems. These two application domains pose certain salient technical challenges. In agriculture the data is often sparse, due to the vast areas of farms and the requirement to keep the cost of monitoring low. Second, in both domains, there are multiple types of sensors with varying capabilities and costs. The sensor data characteristics change with the operating point of the environment or machines, such as, the RPM of the motor. The inferencing and the anomaly detection processes therefore have to be calibrated for the operating point.
  In this paper, we analyze data from sensors deployed in an agricultural farm with data from seven different kinds of sensors, and from an advanced manufacturing testbed with vibration sensors. We evaluate the performance of ARIMA and LSTM models for predicting the time series of sensor data. Then, considering the sparse data from one kind of sensor, we perform transfer learning from a high data rate sensor. We then perform anomaly detection using the predicted sensor data. Taken together, we show how in these two application domains, predictive failure classification can be achieved, thus paving the way for predictive maintenance.
       


### [Interpretable Predictive Maintenance for Hard Drives](https://arxiv.org/abs/2102.06509)

**Authors:**
Maxime Amram, Jack Dunn, Jeremy J. Toledano, Ying Daisy Zhuo

**Abstract:**
Existing machine learning approaches for data-driven predictive maintenance are usually black boxes that claim high predictive power yet cannot be understood by humans. This limits the ability of humans to use these models to derive insights and understanding of the underlying failure mechanisms, and also limits the degree of confidence that can be placed in such a system to perform well on future data. We consider the task of predicting hard drive failure in a data center using recent algorithms for interpretable machine learning. We demonstrate that these methods provide meaningful insights about short- and long-term drive health, while also maintaining high predictive performance. We also show that these analyses still deliver useful insights even when limited historical data is available, enabling their use in situations where data collection has only recently begun.
       


### [Multivariable Fractional Polynomials for lithium-ion batteries degradation models under dynamic conditions](https://arxiv.org/abs/2102.08111)

**Authors:**
Clara Bertinelli Salucci, Azzeddine Bakdi, Ingrid K. Glad, Erik Vanem, Riccardo De Bin

**Abstract:**
Longevity and safety of lithium-ion batteries are facilitated by efficient monitoring and adjustment of the battery operating conditions. Hence, it is crucial to implement fast and accurate algorithms for State of Health (SoH) monitoring on the Battery Management System. The task is challenging due to the complexity and multitude of the factors contributing to the battery degradation, especially because the different degradation processes occur at various timescales and their interactions play an important role. Data-driven methods bypass this issue by approximating the complex processes with statistical or machine learning models. This paper proposes a data-driven approach which is understudied in the context of battery degradation, despite its simplicity and ease of computation: the Multivariable Fractional Polynomial (MFP) regression. Models are trained from historical data of one exhausted cell and used to predict the SoH of other cells. The data are characterised by varying loads simulating dynamic operating conditions. Two hypothetical scenarios are considered: one assumes that a recent capacity measurement is known, the other is based only on the nominal capacity. It was shown that the degradation behaviour of the batteries under examination is influenced by their historical data, as supported by the low prediction errors achieved (root mean squared errors from 1.2% to 7.22% when considering data up to the battery End of Life). Moreover, we offer a multi-factor perspective where the degree of impact of each different factor is analysed. Finally, we compare with a Long Short-Term Memory Neural Network and other works from the literature on the same dataset. We conclude that the MFP regression is effective and competitive with contemporary works, and provides several additional advantages e.g. in terms of interpretability, generalisability, and implementability.
       


### [Genetically Optimized Prediction of Remaining Useful Life](https://arxiv.org/abs/2102.08845)

**Authors:**
Shaashwat Agrawal, Sagnik Sarkar, Gautam Srivastava, Praveen Kumar Reddy Maddikunta, Thippa Reddy Gadekallu

**Abstract:**
The application of remaining useful life (RUL) prediction has taken great importance in terms of energy optimization, cost-effectiveness, and risk mitigation. The existing RUL prediction algorithms mostly constitute deep learning frameworks. In this paper, we implement LSTM and GRU models and compare the obtained results with a proposed genetically trained neural network. The current models solely depend on Adam and SGD for optimization and learning. Although the models have worked well with these optimizers, even little uncertainties in prognostics prediction can result in huge losses. We hope to improve the consistency of the predictions by adding another layer of optimization using Genetic Algorithms. The hyper-parameters - learning rate and batch size are optimized beyond manual capacity. These models and the proposed architecture are tested on the NASA Turbofan Jet Engine dataset. The optimized architecture can predict the given hyper-parameters autonomously and provide superior results.
       


### [Evolving Fuzzy System Applied to Battery Charge Capacity Prediction for Fault Prognostics](https://arxiv.org/abs/2102.09521)

**Authors:**
Murilo Osorio Camargos, Iury Bessa, Luiz A. Q. Cordovil Junior, Pedro Henrique Silva Coutinho, Daniel Furtado Leite, Reinaldo Martinez Palhares

**Abstract:**
This paper addresses the use of data-driven evolving techniques applied to fault prognostics. In such problems, accurate predictions of multiple steps ahead are essential for the Remaining Useful Life (RUL) estimation of a given asset. The fault prognostics' solutions must be able to model the typical nonlinear behavior of the degradation processes of these assets, and be adaptable to each unit's particularities. In this context, the Evolving Fuzzy Systems (EFSs) are models capable of representing such behaviors, in addition of being able to deal with non-stationary behavior, also present in these problems. Moreover, a methodology to recursively track the model's estimation error is presented as a way to quantify uncertainties that are propagated in the long-term predictions. The well-established NASA's Li-ion batteries data set is used to evaluate the models. The experiments indicate that generic EFSs can take advantage of both historical and stream data to estimate the RUL and its uncertainty.
       


### [Neuroscience-Inspired Algorithms for the Predictive Maintenance of Manufacturing Systems](https://arxiv.org/abs/2102.11450)

**Authors:**
Arnav V. Malawade, Nathan D. Costa, Deepan Muthirayan, Pramod P. Khargonekar, Mohammad A. Al Faruque

**Abstract:**
If machine failures can be detected preemptively, then maintenance and repairs can be performed more efficiently, reducing production costs. Many machine learning techniques for performing early failure detection using vibration data have been proposed; however, these methods are often power and data-hungry, susceptible to noise, and require large amounts of data preprocessing. Also, training is usually only performed once before inference, so they do not learn and adapt as the machine ages. Thus, we propose a method of performing online, real-time anomaly detection for predictive maintenance using Hierarchical Temporal Memory (HTM). Inspired by the human neocortex, HTMs learn and adapt continuously and are robust to noise. Using the Numenta Anomaly Benchmark, we empirically demonstrate that our approach outperforms state-of-the-art algorithms at preemptively detecting real-world cases of bearing failures and simulated 3D printer failures. Our approach achieves an average score of 64.71, surpassing state-of-the-art deep-learning (49.38) and statistical (61.06) methods.
       


## March
### [Predictive Maintenance Tool for Non-Intrusive Inspection Systems](https://arxiv.org/abs/2103.01044)

**Authors:**
Georgi Nalbantov, Dimitar Todorov, Nikolay Zografov, Stefan Georgiev, Nadia Bojilova

**Abstract:**
Cross-border security is of topmost priority for societies. Economies lose billions each year due to counterfeiters and other threats. Security checkpoints equipped with X-ray Security Systems (NIIS-Non-Intrusive Inspection Systems) like airports, ports, border control and customs authorities tackle the myriad of threats by using NIIS to inspect bags, air, land, sea and rail cargo, and vehicles. The reliance on the X-ray scanning systems necessitates their continuous 24/7 functioning being provided for. Hence the need for their working condition being closely monitored and preemptive actions being taken to reduce the overall X-ray systems downtime. In this paper, we present a predictive maintenance decision support system, abbreviated as PMT4NIIS (Predictive Maintenance Tool for Non-Intrusive Inspection Systems), which is a kind of augmented analytics platforms that provides real-time AI-generated warnings for upcoming risk of system malfunctioning leading to possible downtime. The industrial platform is the basis of a 24/7 Service Desk and Monitoring center for the working condition of various X-ray Security Systems.
       


### [Multi-Class Multiple Instance Learning for Predicting Precursors to Aviation Safety Events](https://arxiv.org/abs/2103.06244)

**Authors:**
Marc-Henri Bleu-Laine, Tejas G. Puranik, Dimitri N. Mavris, Bryan Matthews

**Abstract:**
In recent years, there has been a rapid growth in the application of machine learning techniques that leverage aviation data collected from commercial airline operations to improve safety. Anomaly detection and predictive maintenance have been the main targets for machine learning applications. However, this paper focuses on the identification of precursors, which is a relatively newer application. Precursors are events correlated with adverse events that happen prior to the adverse event itself. Therefore, precursor mining provides many benefits including understanding the reasons behind a safety incident and the ability to identify signatures, which can be tracked throughout a flight to alert the operators of the potential for an adverse event in the future. This work proposes using the multiple-instance learning (MIL) framework, a weakly supervised learning task, combined with carefully designed binary classifier leveraging a Multi-Head Convolutional Neural Network-Recurrent Neural Network (MHCNN-RNN) architecture. Multi-class classifiers are then created and compared, enabling the prediction of different adverse events for any given flight by combining binary classifiers, and by modifying the MHCNN-RNN to handle multiple outputs. Results obtained showed that the multiple binary classifiers perform better and are able to accurately forecast high speed and high path angle events during the approach phase. Multiple binary classifiers are also capable of determining the aircraft's parameters that are correlated to these events. The identified parameters can be considered precursors to the events and may be studied/tracked further to prevent these events in the future.
       


### [Predictive Maintenance -- Bridging Artificial Intelligence and IoT](https://arxiv.org/abs/2103.11148)

**Authors:**
G. G. Samatas, S. S. Moumgiakmas, G. A. Papakostas

**Abstract:**
This paper highlights the trends in the field of predictive maintenance with the use of machine learning. With the continuous development of the Fourth Industrial Revolution, through IoT, the technologies that use artificial intelligence are evolving. As a result, industries have been using these technologies to optimize their production. Through scientific research conducted for this paper, conclusions were drawn about the trends in Predictive Maintenance applications with the use of machine learning bridging Artificial Intelligence and IoT. These trends are related to the types of industries in which Predictive Maintenance was applied, the models of artificial intelligence were implemented, mainly of machine learning and the types of sensors that are applied through the IoT to the applications. Six sectors were presented and the production sector was dominant as it accounted for 54.54% of total publications. In terms of artificial intelligence models, the most prevalent among ten were the Artificial Neural Networks, Support Vector Machine and Random Forest with 27.84%, 17.72% and 13.92% respectively. Finally, twelve categories of sensors emerged, of which the most widely used were the sensors of temperature and vibration with percentages of 60.71% and 46.42% correspondingly.
       


### [Blockchain-based Digital Twins: Research Trends, Issues, and Future Challenges](https://arxiv.org/abs/2103.11585)

**Authors:**
Sabah Suhail, Rasheed Hussain, Raja Jurdak, Alma Oracevic, Khaled Salah, Raimundas Matulevičius, Choong Seon Hong

**Abstract:**
Industrial processes rely on sensory data for decision-making processes, risk assessment, and performance evaluation. Extracting actionable insights from the collected data calls for an infrastructure that can ensure the dissemination of trustworthy data. For the physical data to be trustworthy, it needs to be cross-validated through multiple sensor sources with overlapping fields of view. Cross-validated data can then be stored on the blockchain, to maintain its integrity and trustworthiness. Once trustworthy data is recorded on the blockchain, product lifecycle events can be fed into data-driven systems for process monitoring, diagnostics, and optimized control. In this regard, Digital Twins (DTs) can be leveraged to draw intelligent conclusions from data by identifying the faults and recommending precautionary measures ahead of critical events. Empowering DTs with blockchain in industrial use-cases targets key challenges of disparate data repositories, untrustworthy data dissemination, and the need for predictive maintenance. In this survey, while highlighting the key benefits of using blockchain-based DTs, we present a comprehensive review of the state-of-the-art research results for blockchain-based DTs. Based on the current research trends, we discuss a trustworthy blockchain-based DTs framework. We highlight the role of Artificial Intelligence (AI) in blockchain-based DTs. Furthermore, we discuss current and future research and deployment challenges of blockchain-supported DTs that require further investigation.
       


### [Adaptive Degradation Process with Deep Learning-Driven Trajectory](https://arxiv.org/abs/2103.11598)

**Author:**
Li Yang

**Abstract:**
Remaining useful life (RUL) estimation is a crucial component in the implementation of intelligent predictive maintenance and health management. Deep neural network (DNN) approaches have been proven effective in RUL estimation due to their capacity in handling high-dimensional non-linear degradation features. However, the applications of DNN in practice face two challenges: (a) online update of lifetime information is often unavailable, and (b) uncertainties in predicted values may not be analytically quantified. This paper addresses these issues by developing a hybrid DNN-based prognostic approach, where a Wiener-based-degradation model is enhanced with adaptive drift to characterize the system degradation. An LSTM-CNN encoder-decoder is developed to predict future degradation trajectories by jointly learning noise coefficients as well as drift coefficients, and adaptive drift is updated via Bayesian inference. A computationally efficient algorithm is proposed for the calculation of RUL distributions. Numerical experiments are presented using turbofan engines degradation data to demonstrate the superior accuracy of RUL prediction of our proposed approach.
       


### [A Dynamic Battery State-of-Health Forecasting Model for Electric Trucks: Li-Ion Batteries Case-Study](https://arxiv.org/abs/2103.16280)

**Authors:**
Matti Huotari, Shashank Arora, Avleen Malhi, Kary Främling

**Abstract:**
It is of extreme importance to monitor and manage the battery health to enhance the performance and decrease the maintenance cost of operating electric vehicles. This paper concerns the machine-learning-enabled state-of-health (SoH) prognosis for Li-ion batteries in electric trucks, where they are used as energy sources. The paper proposes methods to calculate SoH and cycle life for the battery packs. We propose autoregressive integrated modeling average (ARIMA) and supervised learning (bagging with decision tree as the base estimator; BAG) for forecasting the battery SoH in order to maximize the battery availability for forklift operations. As the use of data-driven methods for battery prognostics is increasing, we demonstrate the capabilities of ARIMA and under circumstances when there is little prior information available about the batteries. For this work, we had a unique data set of 31 lithium-ion battery packs from forklifts in commercial operations. On the one hand, results indicate that the developed ARIMA model provided relevant tools to analyze the data from several batteries. On the other hand, BAG model results suggest that the developed supervised learning model using decision trees as base estimator yields better forecast accuracy in the presence of large variation in data for one battery.
       


### [VisioRed: A Visualisation Tool for Interpretable Predictive Maintenance](https://arxiv.org/abs/2103.17003)

**Authors:**
Spyridon Paraschos, Ioannis Mollas, Nick Bassiliades, Grigorios Tsoumakas

**Abstract:**
The use of machine learning rapidly increases in high-risk scenarios where decisions are required, for example in healthcare or industrial monitoring equipment. In crucial situations, a model that can offer meaningful explanations of its decision-making is essential. In industrial facilities, the equipment's well-timed maintenance is vital to ensure continuous operation to prevent money loss. Using machine learning, predictive and prescriptive maintenance attempt to anticipate and prevent eventual system failures. This paper introduces a visualisation tool incorporating interpretations to display information derived from predictive maintenance models, trained on time-series data.
       


## April
### [Uncertainty-aware Remaining Useful Life predictor](https://arxiv.org/abs/2104.03613)

**Authors:**
Luca Biggio, Alexander Wieland, Manuel Arias Chao, Iason Kastanis, Olga Fink

**Abstract:**
Remaining Useful Life (RUL) estimation is the problem of inferring how long a certain industrial asset can be expected to operate within its defined specifications. Deploying successful RUL prediction methods in real-life applications is a prerequisite for the design of intelligent maintenance strategies with the potential of drastically reducing maintenance costs and machine downtimes. In light of their superior performance in a wide range of engineering fields, Machine Learning (ML) algorithms are natural candidates to tackle the challenges involved in the design of intelligent maintenance systems. In particular, given the potentially catastrophic consequences or substantial costs associated with maintenance decisions that are either too late or too early, it is desirable that ML algorithms provide uncertainty estimates alongside their predictions. However, standard data-driven methods used for uncertainty estimation in RUL problems do not scale well to large datasets or are not sufficiently expressive to model the high-dimensional mapping from raw sensor data to RUL estimates. In this work, we consider Deep Gaussian Processes (DGPs) as possible solutions to the aforementioned limitations. We perform a thorough evaluation and comparison of several variants of DGPs applied to RUL predictions. The performance of the algorithms is evaluated on the N-CMAPSS (New Commercial Modular Aero-Propulsion System Simulation) dataset from NASA for aircraft engines. The results show that the proposed methods are able to provide very accurate RUL predictions along with sensible uncertainty estimates, providing more reliable solutions for (safety-critical) real-life industrial applications.
       


### [Learning representations with end-to-end models for improved remaining useful life prognostics](https://arxiv.org/abs/2104.05049)

**Authors:**
Alaaeddine Chaoub, Alexandre Voisin, Christophe Cerisara, Benoît Iung

**Abstract:**
The remaining Useful Life (RUL) of equipment is defined as the duration between the current time and its failure. An accurate and reliable prognostic of the remaining useful life provides decision-makers with valuable information to adopt an appropriate maintenance strategy to maximize equipment utilization and avoid costly breakdowns. In this work, we propose an end-to-end deep learning model based on multi-layer perceptron and long short-term memory layers (LSTM) to predict the RUL. After normalization of all data, inputs are fed directly to an MLP layers for feature learning, then to an LSTM layer to capture temporal dependencies, and finally to other MLP layers for RUL prognostic. The proposed architecture is tested on the NASA commercial modular aero-propulsion system simulation (C-MAPSS) dataset. Despite its simplicity with respect to other recently proposed models, the model developed outperforms them with a significant decrease in the competition score and in the root mean square error score between the predicted and the gold value of the RUL. In this paper, we will discuss how the proposed end-to-end model is able to achieve such good results and compare it to other deep learning and state-of-the-art methods.
       


### [Maintenance scheduling of manufacturing systems based on optimal price of the network](https://arxiv.org/abs/2104.06654)

**Authors:**
Pegah Rokhforoz, Olga Fink

**Abstract:**
Goods can exhibit positive externalities impacting decisions of customers in socials networks. Suppliers can integrate these externalities in their pricing strategies to increase their revenue. Besides optimizing the prize, suppliers also have to consider their production and maintenance costs. Predictive maintenance has the potential to reduce the maintenance costs and improve the system availability. To address the joint optimization of pricing with network externalities and predictive maintenance scheduling based on the condition of the system, we propose a bi-level optimization solution based on game theory. In the first level, the manufacturing company decides about the predictive maintenance scheduling of the units and the price of the goods. In the second level, the customers decide about their consumption using an optimization approach in which the objective function depends on their consumption, the consumption levels of other customers who are connected through the graph, and the price of the network which is determined by the supplier. To solve the problem, we propose the leader-multiple-followers game where the supplier as a leader predicts the strategies of the followers. Then, customers as the followers obtain their strategies based on the leader's and other followers' strategies. We demonstrate the effectiveness of our proposed method on a simulated case study. The results demonstrate that knowledge of the social network graph results in an increased revenue compared to the case when the underlying social network graph is not known. Moreover, the results demonstrate that obtaining the predictive maintenance scheduling based on the proposed optimization approach leads to an increased profit compared to the baseline decision-making (perform maintenance at the degradation limit).
       


## May
### [Autoregressive Hidden Markov Models with partial knowledge on latent space applied to aero-engines prognostics](https://arxiv.org/abs/2105.00211)

**Authors:**
Pablo Juesas, Emmanuel Ramasso, Sébastien Drujont, Vincent Placet

**Abstract:**
[This paper was initially published in PHME conference in 2016, selected for further publication in International Journal of Prognostics and Health Management.]
  This paper describes an Autoregressive Partially-hidden Markov model (ARPHMM) for fault detection and prognostics of equipments based on sensors' data. It is a particular dynamic Bayesian network that allows to represent the dynamics of a system by means of a Hidden Markov Model (HMM) and an autoregressive (AR) process. The Markov chain assumes that the system is switching back and forth between internal states while the AR process ensures a temporal coherence on sensor measurements. A sound learning procedure of standard ARHMM based on maximum likelihood allows to iteratively estimate all parameters simultaneously. This paper suggests a modification of the learning procedure considering that one may have prior knowledge about the structure which becomes partially hidden. The integration of the prior is based on the Theory of Weighted Distributions which is compatible with the Expectation-Maximization algorithm in the sense that the convergence properties are still satisfied. We show how to apply this model to estimate the remaining useful life based on health indicators. The autoregressive parameters can indeed be used for prediction while the latent structure can be used to get information about the degradation level. The interest of the proposed method for prognostics and health assessment is demonstrated on CMAPSS datasets.
       


### [Enhancing Generalizability of Predictive Models with Synergy of Data and Physics](https://arxiv.org/abs/2105.01429)

**Authors:**
Yingjun Shen, Zhe Song, Andrew Kusiak

**Abstract:**
Wind farm needs prediction models for predictive maintenance. There is a need to predict values of non-observable parameters beyond ranges reflected in available data. A prediction model developed for one machine many not perform well in another similar machine. This is usually due to lack of generalizability of data-driven models. To increase generalizability of predictive models, this research integrates the data mining with first-principle knowledge. Physics-based principles are combined with machine learning algorithms through feature engineering, strong rules and divide-and-conquer. The proposed synergy concept is illustrated with the wind turbine blade icing prediction and achieves significant prediction accuracy across different turbines. The proposed process is widely accepted by wind energy predictive maintenance practitioners because of its simplicity and efficiency. Furthermore, this paper demonstrates the importance of embedding physical principles within the machine learning process, and also highlight an important point that the need for more complex machine learning algorithms in industrial big data mining is often much less than it is in other applications, making it essential to incorporate physics and follow Less is More philosophy.
       


### [Evaluating the Effect of Longitudinal Dose and INR Data on Maintenance Warfarin Dose Predictions](https://arxiv.org/abs/2105.02625)

**Authors:**
Anish Karpurapu, Adam Krekorian, Ye Tian, Leslie M. Collins, Ravi Karra, Aaron Franklin, Boyla O. Mainsah

**Abstract:**
Warfarin, a commonly prescribed drug to prevent blood clots, has a highly variable individual response. Determining a maintenance warfarin dose that achieves a therapeutic blood clotting time, as measured by the international normalized ratio (INR), is crucial in preventing complications. Machine learning algorithms are increasingly being used for warfarin dosing; usually, an initial dose is predicted with clinical and genotype factors, and this dose is revised after a few days based on previous doses and current INR. Since a sequence of prior doses and INR better capture the variability in individual warfarin response, we hypothesized that longitudinal dose response data will improve maintenance dose predictions. To test this hypothesis, we analyzed a dataset from the COAG warfarin dosing study, which includes clinical data, warfarin doses and INR measurements over the study period, and maintenance dose when therapeutic INR was achieved. Various machine learning regression models to predict maintenance warfarin dose were trained with clinical factors and dosing history and INR data as features. Overall, dose revision algorithms with a single dose and INR achieved comparable performance as the baseline dose revision algorithm. In contrast, dose revision algorithms with longitudinal dose and INR data provided maintenance dose predictions that were statistically significantly much closer to the true maintenance dose. Focusing on the best performing model, gradient boosting (GB), the proportion of ideal estimated dose, i.e., defined as within $\pm$20% of the true dose, increased from the baseline (54.92%) to the GB model with the single (63.11%) and longitudinal (75.41%) INR. More accurate maintenance dose predictions with longitudinal dose response data can potentially achieve therapeutic INR faster, reduce drug-related complications and improve patient outcomes with warfarin therapy.
       


### [A Computational Framework for Modeling Complex Sensor Network Data Using Graph Signal Processing and Graph Neural Networks in Structural Health Monitoring](https://arxiv.org/abs/2105.05316)

**Authors:**
Stefan Bloemheuvel, Jurgen van den Hoogen, Martin Atzmueller

**Abstract:**
Complex networks lend themselves to the modeling of multidimensional data, such as relational and/or temporal data. In particular, when such complex data and their inherent relationships need to be formalized, complex network modeling and its resulting graph representations enable a wide range of powerful options. In this paper, we target this - connected to specific machine learning approaches on graphs for structural health monitoring on an analysis and predictive (maintenance) perspective. Specifically, we present a framework based on Complex Network Modeling, integrating Graph Signal Processing (GSP) and Graph Neural Network (GNN) approaches. We demonstrate this framework in our targeted application domain of Structural Health Monitoring (SHM). In particular, we focus on a prominent real-world structural health monitoring use case, i.e., modeling and analyzing sensor data (strain, vibration) of a large bridge in the Netherlands. In our experiments, we show that GSP enables the identification of the most important sensors, for which we investigate a set of search and optimization approaches. Furthermore, GSP enables the detection of specific graph signal patterns (mode shapes), capturing physical functional properties of the sensors in the applied complex network. In addition, we show the efficacy of applying GNNs for strain prediction on this kind of data.
       


### [Some Challenges in Monitoring Epidemics](https://arxiv.org/abs/2105.08384)

**Authors:**
Vaiva Vasiliauskaite, Nino Antulov-Fantulin, Dirk Helbing

**Abstract:**
Epidemic models often reflect characteristic features of infectious spreading processes by coupled non-linear differential equations considering different states of health (such as Susceptible, Infected, or Recovered). This compartmental modeling approach, however, delivers an incomplete picture of the dynamics of epidemics, as it neglects stochastic and network effects, and also the role of the measurement process, on which the estimation of epidemiological parameters and incidence values relies. In order to study the related issues, we extend established epidemiological spreading models with a model of the measurement (i.e. testing) process, considering the problems of false positives and false negatives as well as biased sampling. Studying a model-generated ground truth in conjunction with simulated observation processes (virtual measurements) allows one to gain insights into the limitations of purely data-driven methods to assess the epidemic situation. We conclude that epidemic monitoring, simulation, and forecasting are wicked problems, as applying a conventional data-driven approach to a complex system with non-linear dynamics, network effects, and uncertainty can be misleading. Nevertheless, some of the errors can be corrected for, using scientific knowledge of the spreading dynamics and the measurement process. We conclude that such corrections should generally be part of epidemic monitoring, modeling, and forecasting efforts.
       


### [CONECT4: Desarrollo de componentes basados en Realidad Mixta, Realidad Virtual Y Conocimiento Experto para generación de entornos de aprendizaje Hombre-Máquina](https://arxiv.org/abs/2105.11216)

**Authors:**
Santiago González, Alvaro García, Ana Núñez

**Abstract:**
This work presents the results of project CONECT4, which addresses the research and development of new non-intrusive communication methods for the generation of a human-machine learning ecosystem oriented to predictive maintenance in the automotive industry. Through the use of innovative technologies such as Augmented Reality, Virtual Reality, Digital Twin and expert knowledge, CONECT4 implements methodologies that allow improving the efficiency of training techniques and knowledge management in industrial companies. The research has been supported by the development of content and systems with a low level of technological maturity that address solutions for the industrial sector applied in training and assistance to the operator. The results have been analyzed in companies in the automotive sector, however, they are exportable to any other type of industrial sector. -- --
  En esta publicación se presentan los resultados del proyecto CONECT4, que aborda la investigación y desarrollo de nuevos métodos de comunicación no intrusivos para la generación de un ecosistema de aprendizaje hombre-máquina orientado al mantenimiento predictivo en la industria de automoción. A través del uso de tecnologías innovadoras como la Realidad Aumentada, la Realidad Virtual, el Gemelo Digital y conocimiento experto, CONECT4 implementa metodologías que permiten mejorar la eficiencia de las técnicas de formación y gestión de conocimiento en las empresas industriales. La investigación se ha apoyado en el desarrollo de contenidos y sistemas con un nivel de madurez tecnológico bajo que abordan soluciones para el sector industrial aplicadas en la formación y asistencia al operario. Los resultados han sido analizados en empresas del sector de automoción, no obstante, son exportables a cualquier otro tipo de sector industrial.
       


### [Anomaly Detection in Predictive Maintenance: A New Evaluation Framework for Temporal Unsupervised Anomaly Detection Algorithms](https://arxiv.org/abs/2105.12818)

**Authors:**
Jacinto Carrasco, Irina Markova, David López, Ignacio Aguilera, Diego García, Marta García-Barzana, Manuel Arias-Rodil, Julián Luengo, Francisco Herrera

**Abstract:**
The research in anomaly detection lacks a unified definition of what represents an anomalous instance. Discrepancies in the nature itself of an anomaly lead to multiple paradigms of algorithms design and experimentation. Predictive maintenance is a special case, where the anomaly represents a failure that must be prevented. Related time-series research as outlier and novelty detection or time-series classification does not apply to the concept of an anomaly in this field, because they are not single points which have not been seen previously and may not be precisely annotated. Moreover, due to the lack of annotated anomalous data, many benchmarks are adapted from supervised scenarios.
  To address these issues, we generalise the concept of positive and negative instances to intervals to be able to evaluate unsupervised anomaly detection algorithms. We also preserve the imbalance scheme for evaluation through the proposal of the Preceding Window ROC, a generalisation for the calculation of ROC curves for time-series scenarios. We also adapt the mechanism from a established time-series anomaly detection benchmark to the proposed generalisations to reward early detection. Therefore, the proposal represents a flexible evaluation framework for the different scenarios. To show the usefulness of this definition, we include a case study of Big Data algorithms with a real-world time-series problem provided by the company ArcelorMittal, and compare the proposal with an evaluation method.
       


## June
### [Online Detection of Vibration Anomalies Using Balanced Spiking Neural Networks](https://arxiv.org/abs/2106.00687)

**Authors:**
Nik Dennler, Germain Haessig, Matteo Cartiglia, Giacomo Indiveri

**Abstract:**
Vibration patterns yield valuable information about the health state of a running machine, which is commonly exploited in predictive maintenance tasks for large industrial systems. However, the overhead, in terms of size, complexity and power budget, required by classical methods to exploit this information is often prohibitive for smaller-scale applications such as autonomous cars, drones or robotics. Here we propose a neuromorphic approach to perform vibration analysis using spiking neural networks that can be applied to a wide range of scenarios. We present a spike-based end-to-end pipeline able to detect system anomalies from vibration data, using building blocks that are compatible with analog-digital neuromorphic circuits. This pipeline operates in an online unsupervised fashion, and relies on a cochlea model, on feedback adaptation and on a balanced spiking neural network. We show that the proposed method achieves state-of-the-art performance or better against two publicly available data sets. Further, we demonstrate a working proof-of-concept implemented on an asynchronous neuromorphic processor device. This work represents a significant step towards the design and implementation of autonomous low-power edge-computing devices for online vibration monitoring.
       


### [Short-term Maintenance Planning of Autonomous Trucks for Minimizing Economic Risk](https://arxiv.org/abs/2106.01871)

**Authors:**
Xin Tao, Jonas Mårtensson, Håkan Warnquist, Anna Pernestål

**Abstract:**
New autonomous driving technologies are emerging every day and some of them have been commercially applied in the real world. While benefiting from these technologies, autonomous trucks are facing new challenges in short-term maintenance planning, which directly influences the truck operator's profit. In this paper, we implement a vehicle health management system by addressing the maintenance planning issues of autonomous trucks on a transport mission. We also present a maintenance planning model using a risk-based decision-making method, which identifies the maintenance decision with minimal economic risk of the truck company. Both availability losses and maintenance costs are considered when evaluating the economic risk. We demonstrate the proposed model by numerical experiments illustrating real-world scenarios. In the experiments, compared to three baseline methods, the expected economic risk of the proposed method is reduced by up to $47\%$. We also conduct sensitivity analyses of different model parameters. The analyses show that the economic risk significantly decreases when the estimation accuracy of remaining useful life, the maximal allowed time of delivery delay before order cancellation, or the number of workshops increases. The experiment results contribute to identifying future research and development attentions of autonomous trucks from an economic perspective.
       


### [Collection and harmonization of system logs and prototypal Analytics services with the Elastic (ELK) suite at the INFN-CNAF computing centre](https://arxiv.org/abs/2106.02612)

**Authors:**
Tommaso Diotalevi, Antonio Falabella, Barbara Martelli, Diego Michelotto, Lucia Morganti, Daniele Bonacorsi, Luca Giommi, Simone Rossi Tisbeni

**Abstract:**
The distributed Grid infrastructure for High Energy Physics experiments at the Large Hadron Collider (LHC) in Geneva comprises a set of computing centres, spread all over the world, as part of the Worldwide LHC Computing Grid (WLCG). In Italy, the Tier-1 functionalities are served by the INFN-CNAF data center, which provides also computing and storage resources to more than twenty non-LHC experiments. For this reason, a high amount of logs are collected each day from various sources, which are highly heterogeneous and difficult to harmonize. In this contribution, a working implementation of a system that collects, parses and displays the log information from CNAF data sources and the investigation of a Machine Learning based predictive maintenance system, is presented.
       


### [iThing: Designing Next-Generation Things with Battery Health Self-Monitoring Capabilities for Sustainable IoT in Smart Cities](https://arxiv.org/abs/2106.06678)

**Authors:**
Aparna Sinha, Debanjan Das, Venkanna Udutalapally, Mukil Kumar Selvarajan, Saraju P. Mohanty

**Abstract:**
An accurate and reliable technique for predicting Remaining Useful Life (RUL) for battery cells proves helpful in battery-operated IoT devices, especially in remotely operated sensor nodes. Data-driven methods have proved to be the most effective methods until now. These IoT devices have low computational capabilities to save costs, but Data-Driven battery health techniques often require a comparatively large amount of computational power to predict SOH and RUL due to most methods being feature-heavy. This issue calls for ways to predict RUL with the least amount of calculations and memory. This paper proposes an effective and novel peak extraction method to reduce computation and memory needs and provide accurate prediction methods using the least number of features while performing all calculations on-board. The model can self-sustain, requires minimal external interference, and hence operate remotely much longer. Experimental results prove the accuracy and reliability of this method. The Absolute Error (AE), Relative error (RE), and Root Mean Square Error (RMSE) are calculated to compare effectiveness. The training of the GPR model takes less than 2 seconds, and the correlation between SOH from peak extraction and RUL is 0.97.
       


### [Intelligent Vision Based Wear Forecasting on Surfaces of Machine Tool Elements](https://arxiv.org/abs/2106.06839)

**Authors:**
Tobias Schlagenhauf, Niklas Burghardt

**Abstract:**
This paper addresses the ability to enable machines to automatically detect failures on machine tool components as well as estimating the severity of the failures, which is a critical step towards autonomous production machines. Extracting information about the severity of failures has been a substantial part of classical, as well as Machine Learning based machine vision systems. Efforts have been undertaken to automatically predict the severity of failures on machine tool components for predictive maintenance purposes. Though, most approaches only partly cover a completely automatic system from detecting failures to the prognosis of their future severity. To the best of the authors knowledge, this is the first time a vision-based system for defect detection and prognosis of failures on metallic surfaces in general and on Ball Screw Drives in specific has been proposed. The authors show that they can do both, detect and prognose the evolution of a failure on the surface of a Ball Screw Drive.
       


### [Certification of embedded systems based on Machine Learning: A survey](https://arxiv.org/abs/2106.07221)

**Authors:**
Guillaume Vidot, Christophe Gabreau, Ileana Ober, Iulian Ober

**Abstract:**
Advances in machine learning (ML) open the way to innovating functions in the avionic domain, such as navigation/surveillance assistance (e.g. vision-based navigation, obstacle sensing, virtual sensing), speechto-text applications, autonomous flight, predictive maintenance or cockpit assistance. Current certification standards and practices, which were defined and refined decades over decades with classical programming in mind, do not however support this new development paradigm. This article provides an overview of the main challenges raised by the use ML in the demonstration of compliance with regulation requirements, and a survey of literature relevant to these challenges, with particular focus on the issues of robustness and explainability of ML results.
       


### [Towards the Objective Speech Assessment of Smoking Status based on Voice Features: A Review of the Literature](https://arxiv.org/abs/2106.07874)

**Authors:**
Zhizhong Ma, Chris Bullen, Joanna Ting Wai Chu, Ruili Wang, Yingchun Wang, Satwinder Singh

**Abstract:**
In smoking cessation clinical research and practice, objective validation of self-reported smoking status is crucial for ensuring the reliability of the primary outcome, that is, smoking abstinence. Speech signals convey important information about a speaker, such as age, gender, body size, emotional state, and health state. We investigated (1) if smoking could measurably alter voice features, (2) if smoking cessation could lead to changes in voice, and therefore (3) if the voice-based smoking status assessment has the potential to be used as an objective smoking cessation validation method.
       


### [Labelling Drifts in a Fault Detection System for Wind Turbine Maintenance](https://arxiv.org/abs/2106.09951)

**Authors:**
Iñigo Martinez, Elisabeth Viles, Iñaki Cabrejas

**Abstract:**
A failure detection system is the first step towards predictive maintenance strategies. A popular data-driven method to detect incipient failures and anomalies is the training of normal behaviour models by applying a machine learning technique like feed-forward neural networks (FFNN) or extreme learning machines (ELM). However, the performance of any of these modelling techniques can be deteriorated by the unexpected rise of non-stationarities in the dynamic environment in which industrial assets operate. This unpredictable statistical change in the measured variable is known as concept drift. In this article a wind turbine maintenance case is presented, where non-stationarities of various kinds can happen unexpectedly. Such concept drift events are desired to be detected by means of statistical detectors and window-based approaches. However, in real complex systems, concept drifts are not as clear and evident as in artificially generated datasets. In order to evaluate the effectiveness of current drift detectors and also to design an appropriate novel technique for this specific industrial application, it is essential to dispose beforehand of a characterization of the existent drifts. Under the lack of information in this regard, a methodology for labelling concept drift events in the lifetime of wind turbines is proposed. This methodology will facilitate the creation of a drift database that will serve both as a training ground for concept drift detectors and as a valuable information to enhance the knowledge about maintenance of complex systems.
       


### [Lite-Sparse Hierarchical Partial Power Processing for Second-Use Battery Energy Storage Systems](https://arxiv.org/abs/2106.11749)

**Authors:**
Xiaofan Cui, Alireza Ramyar, Peyman Mohtat, Veronica Contreras, Jason Siegel, Anna Stefanopoulou, Al-Thaddeus Avestruz

**Abstract:**
The explosive growth of electric vehicles (EVs) is leading to a surge in retired EV batteries, which are typically recycled despite having nearly 80% available capacity. Repurposing automotive batteries for second-use battery energy storage systems (2-BESS) has both economical and environmental benefits. The challenge with second-use batteries is the heterogeneity in their state of health. This paper introduces a new strategy to optimize 2-BESS performance despite the heterogeneity of individual batteries while reducing the cost of power conversion. In this paper, the statistical distribution of the power heterogeneity in the supply of batteries is used to optimize the choice of power converters and design the power flow within the battery energy storage system (BESS) to optimize power capability. By leveraging a new lite-sparse hierarchical partial power processing (LS-HiPPP) approach, we study how a hierarchy in partial power processing (PPP) partitions power converters to significantly reduce converter ratings, process less power to achieve high system efficiency with lower cost (lower efficiency) converters, and take advantage of economies of scale by requiring only a minimal number of sets of identical converters. Our results demonstrate that LS-HiPPP architectures offer the best tradeoff between battery utilization and converter cost and have higher system efficiency than conventional partial power processing (C-PPP) in all cases.
       


### [Dual Aspect Self-Attention based on Transformer for Remaining Useful Life Prediction](https://arxiv.org/abs/2106.15842)

**Authors:**
Zhizheng Zhang, Wen Song, Qiqiang Li

**Abstract:**
Remaining useful life prediction (RUL) is one of the key technologies of condition-based maintenance, which is important to maintain the reliability and safety of industrial equipments. Massive industrial measurement data has effectively improved the performance of the data-driven based RUL prediction method. While deep learning has achieved great success in RUL prediction, existing methods have difficulties in processing long sequences and extracting information from the sensor and time step aspects. In this paper, we propose Dual Aspect Self-attention based on Transformer (DAST), a novel deep RUL prediction method, which is an encoder-decoder structure purely based on self-attention without any RNN/CNN module. DAST consists of two encoders, which work in parallel to simultaneously extract features of different sensors and time steps. Solely based on self-attention, the DAST encoders are more effective in processing long data sequences, and are capable of adaptively learning to focus on more important parts of input. Moreover, the parallel feature extraction design avoids mutual influence of information from two aspects. Experiments on two widely used turbofan engines datasets show that our method significantly outperforms the state-of-the-art RUL prediction methods.
       


## July
### [Uncertainty-Aware Learning for Improvements in Image Quality of the Canada-France-Hawaii Telescope](https://arxiv.org/abs/2107.00048)

**Authors:**
Sankalp Gilda, Stark C. Draper, Sebastien Fabbro, William Mahoney, Simon Prunet, Kanoa Withington, Matthew Wilson, Yuan-Sen Ting, Andrew Sheinis

**Abstract:**
We leverage state-of-the-art machine learning methods and a decade's worth of archival data from CFHT to predict observatory image quality (IQ) from environmental conditions and observatory operating parameters. Specifically, we develop accurate and interpretable models of the complex dependence between data features and observed IQ for CFHT's wide-field camera, MegaCam. Our contributions are several-fold. First, we collect, collate and reprocess several disparate data sets gathered by CFHT scientists. Second, we predict probability distribution functions (PDFs) of IQ and achieve a mean absolute error of $\sim0.07''$ for the predicted medians. Third, we explore the data-driven actuation of the 12 dome "vents" installed in 2013-14 to accelerate the flushing of hot air from the dome. We leverage epistemic and aleatoric uncertainties in conjunction with probabilistic generative modeling to identify candidate vent adjustments that are in-distribution (ID); for the optimal configuration for each ID sample, we predict the reduction in required observing time to achieve a fixed SNR. On average, the reduction is $\sim12\%$. Finally, we rank input features by their Shapley values to identify the most predictive variables for each observation. Our long-term goal is to construct reliable and real-time models that can forecast optimal observatory operating parameters to optimize IQ. We can then feed such forecasts into scheduling protocols and predictive maintenance routines. We anticipate that such approaches will become standard in automating observatory operations and maintenance by the time CFHT's successor, the Maunakea Spectroscopic Explorer, is installed in the next decade.
       


### [One-class Steel Detector Using Patch GAN Discriminator for Visualising Anomalous Feature Map](https://arxiv.org/abs/2107.00143)

**Authors:**
Takato Yasuno, Junichiro Fujii, Sakura Fukami

**Abstract:**
For steel product manufacturing in indoor factories, steel defect detection is important for quality control. For example, a steel sheet is extremely delicate, and must be accurately inspected. However, to maintain the painted steel parts of the infrastructure around a severe outdoor environment, corrosion detection is critical for predictive maintenance. In this paper, we propose a general-purpose application for steel anomaly detection that consists of the following four components. The first, a learner, is a unit image classification network to determine whether the region of interest or background has been recognised, after dividing the original large sized image into 256 square unit images. The second, an extractor, is a discriminator feature encoder based on a pre-trained steel generator with a patch generative adversarial network discriminator(GAN). The third, an anomaly detector, is a one-class support vector machine(SVM) to predict the anomaly score using the discriminator feature. The fourth, an indicator, is an anomalous probability map used to visually explain the anomalous features. Furthermore, we demonstrated our method through the inspection of steel sheet defects with 13,774 unit images using high-speed cameras, and painted steel corrosion with 19,766 unit images based on an eye inspection of the photographs. Finally, we visualise anomalous feature maps of steel using a strip and painted steel inspection dataset
       


### [Supporting AI Engineering on the IoT Edge through Model-Driven TinyML](https://arxiv.org/abs/2107.02690)

**Authors:**
Armin Moin, Moharram Challenger, Atta Badii, Stephan Günnemann

**Abstract:**
Software engineering of network-centric Artificial Intelligence (AI) and Internet of Things (IoT) enabled Cyber-Physical Systems (CPS) and services, involves complex design and validation challenges. In this paper, we propose a novel approach, based on the model-driven software engineering paradigm, in particular the domain-specific modeling methodology. We focus on a sub-discipline of AI, namely Machine Learning (ML) and propose the delegation of data analytics and ML to the IoT edge. This way, we may increase the service quality of ML, for example, its availability and performance, regardless of the network conditions, as well as maintaining the privacy, security and sustainability. We let practitioners assign ML tasks to heterogeneous edge devices, including highly resource-constrained embedded microcontrollers with main memories in the order of Kilobytes, and energy consumption in the order of milliwatts. This is known as TinyML. Furthermore, we show how software models with different levels of abstraction, namely platform-independent and platform-specific models can be used in the software development process. Finally, we validate the proposed approach using a case study addressing the predictive maintenance of a hydraulics system with various networked sensors and actuators.
       


### [Comparing seven methods for state-of-health time series prediction for the lithium-ion battery packs of forklifts](https://arxiv.org/abs/2107.05489)

**Authors:**
Matti Huotari, Shashank Arora, Avleen Malhi, Kary Främling

**Abstract:**
A key aspect for the forklifts is the state-of-health (SoH) assessment to ensure the safety and the reliability of uninterrupted power source. Forecasting the battery SoH well is imperative to enable preventive maintenance and hence to reduce the costs. This paper demonstrates the capabilities of gradient boosting regression for predicting the SoH timeseries under circumstances when there is little prior information available about the batteries. We compared the gradient boosting method with light gradient boosting, extra trees, extreme gradient boosting, random forests, long short-term memory networks and with combined convolutional neural network and long short-term memory networks methods. We used multiple predictors and lagged target signal decomposition results as additional predictors and compared the yielded prediction results with different sets of predictors for each method. For this work, we are in possession of a unique data set of 45 lithium-ion battery packs with large variation in the data. The best model that we derived was validated by a novel walk-forward algorithm that also calculates point-wise confidence intervals for the predictions; we yielded reasonable predictions and confidence intervals for the predictions. Furthermore, we verified this model against five other lithium-ion battery packs; the best model generalised to greater extent to this set of battery packs. The results about the final model suggest that we were able to enhance the results in respect to previously developed models. Moreover, we further validated the model for extracting cycle counts presented in our previous work with data from new forklifts; their battery packs completed around 3000 cycles in a 10-year service period, which corresponds to the cycle life for commercial Nickel-Cobalt-Manganese (NMC) cells.
       


### [Reuse of Semantic Models for Emerging Smart Grids Applications](https://arxiv.org/abs/2107.06999)

**Authors:**
Valentina Janev, Dušan Popadić, Dea Pujić, Maria Esther Vidal, Kemele Endris

**Abstract:**
Data in the energy domain grows at unprecedented rates. Despite the great potential that IoT platforms and other big data-driven technologies have brought in the energy sector, data exchange and data integration are still not wholly achieved. As a result, fragmented applications are developed against energy data silos, and data exchange is limited to few applications. Therefore, this paper identifies semantic models that can be reused for building interoperable energy management services and applications. The ambition is to innovate the Institute Mihajlo Pupin proprietary SCADA system and to enable integration of the Institute Mihajlo Pupin services and applications in the European Union (EU) Energy Data Space. The selection of reusable models has been done based on a set of scenarios related to electricity balancing services, predictive maintenance services, and services for the residential, commercial and industrial sectors.
       


### [Jarvis for Aeroengine Analytics: A Speech Enhanced Virtual Reality Demonstrator Based on Mining Knowledge Databases](https://arxiv.org/abs/2107.13403)

**Authors:**
Sławomir Konrad Tadeja, Krzysztof Kutt, Yupu Lu, Pranay Seshadri, Grzegorz J. Nalepa, Per Ola Kristensson

**Abstract:**
In this paper, we present a Virtual Reality (VR) based environment where the engineer interacts with incoming data from a fleet of aeroengines. This data takes the form of 3D computer-aided design (CAD) engine models coupled with characteristic plots for the subsystems of each engine. Both the plots and models can be interacted with and manipulated using speech or gestural input. The characteristic data is ported to a knowledge-based system underpinned by a knowledge-graph storing complex domain knowledge. This permits the system to respond to queries about the current state and health of each aeroengine asset. Responses to these questions require some degree of analysis, which is handled by a semantic knowledge representation layer managing information on aeroengine subsystems. This paper represents a significant step forward for aeroengine analysis in a bespoke VR environment and brings us a step closer to a Jarvis-like system for aeroengine analytics.
       


## August
### [Interpretable Summaries of Black Box Incident Triaging with Subgroup Discovery](https://arxiv.org/abs/2108.03013)

**Authors:**
Youcef Remil, Anes Bendimerad, Marc Plantevit, Céline Robardet, Mehdi Kaytoue

**Abstract:**
The need of predictive maintenance comes with an increasing number of incidents reported by monitoring systems and equipment/software users. In the front line, on-call engineers (OCEs) have to quickly assess the degree of severity of an incident and decide which service to contact for corrective actions. To automate these decisions, several predictive models have been proposed, but the most efficient models are opaque (say, black box), strongly limiting their adoption. In this paper, we propose an efficient black box model based on 170K incidents reported to our company over the last 7 years and emphasize on the need of automating triage when incidents are massively reported on thousands of servers running our product, an ERP. Recent developments in eXplainable Artificial Intelligence (XAI) help in providing global explanations to the model, but also, and most importantly, with local explanations for each model prediction/outcome. Sadly, providing a human with an explanation for each outcome is not conceivable when dealing with an important number of daily predictions. To address this problem, we propose an original data-mining method rooted in Subgroup Discovery, a pattern mining technique with the natural ability to group objects that share similar explanations of their black box predictions and provide a description for each group. We evaluate this approach and present our preliminary results which give us good hope towards an effective OCE's adoption. We believe that this approach provides a new way to address the problem of model agnostic outcome explanation.
       


### [Concept Drift Detection with Variable Interaction Networks](https://arxiv.org/abs/2108.03273)

**Authors:**
Jan Zenisek, Gabriel Kronberger, Josef Wolfartsberger, Norbert Wild, Michael Affenzeller

**Abstract:**
The current development of today's production industry towards seamless sensor-based monitoring is paving the way for concepts such as Predictive Maintenance. By this means, the condition of plants and products in future production lines will be continuously analyzed with the objective to predict any kind of breakdown and trigger preventing actions proactively. Such ambitious predictions are commonly performed with support of machine learning algorithms. In this work, we utilize these algorithms to model complex systems, such as production plants, by focusing on their variable interactions. The core of this contribution is a sliding window based algorithm, designed to detect changes of the identified interactions, which might indicate beginning malfunctions in the context of a monitored production plant. Besides a detailed description of the algorithm, we present results from experiments with a synthetic dynamical system, simulating stable and drifting system behavior.
       


### [Extracting Semantics from Maintenance Records](https://arxiv.org/abs/2108.05454)

**Authors:**
Sharad Dixit, Varish Mulwad, Abhinav Saxena

**Abstract:**
Rapid progress in natural language processing has led to its utilization in a variety of industrial and enterprise settings, including in its use for information extraction, specifically named entity recognition and relation extraction, from documents such as engineering manuals and field maintenance reports. While named entity recognition is a well-studied problem, existing state-of-the-art approaches require large labelled datasets which are hard to acquire for sensitive data such as maintenance records. Further, industrial domain experts tend to distrust results from black box machine learning models, especially when the extracted information is used in downstream predictive maintenance analytics. We overcome these challenges by developing three approaches built on the foundation of domain expert knowledge captured in dictionaries and ontologies. We develop a syntactic and semantic rules-based approach and an approach leveraging a pre-trained language model, fine-tuned for a question-answering task on top of our base dictionary lookup to extract entities of interest from maintenance records. We also develop a preliminary ontology to represent and capture the semantics of maintenance records. Our evaluations on a real-world aviation maintenance records dataset show promising results and help identify challenges specific to named entity recognition in the context of noisy industrial data.
       


### [Power transformer faults diagnosis using undestructive methods (Roger and IEC) and artificial neural network for dissolved gas analysis applied on the functional transformer in the Algerian north-eastern: a comparative study](https://arxiv.org/abs/2108.10205)

**Authors:**
Bouchaoui Lahcene, Kamel Eddine Hemsas, Hacene Mellah, saad eddine benlahneche

**Abstract:**
Nowadays, power transformer aging and failures are viewed with great attention in power transmission industry. Dissolved gas analysis (DGA) is classified among the biggest widely used methods used within the context of asset management policy to detect the incipient faults in their earlier stage in power transformers. Up to now, several procedures have been employed for the lecture of DGA results. Among these useful means, we find Key Gases, Rogers Ratios, IEC Ratios, the historical technique less used today Doernenburg Ratios, the two types of Duval Pentagons methods, several versions of the Duval Triangles method and Logarithmic Nomograph. Problem. DGA data extracted from different units in service served to verify the ability and reliability of these methods in assessing the state of health of the power transformer. Aim. An improving the quality of diagnostics of electrical power transformer by artificial neural network tools based on two conventional methods in the case of a functional power transformer at Sétif province in East North of Algeria. Methodology. Design an inelegant tool for power transformer diagnosis using neural networks based on traditional methods IEC and Rogers, which allows to early detection faults, to increase the reliability, of the entire electrical energy system from transport to consumers and improve a continuity and quality of service. Results. The solution of the problem was carried out by using feed-forward back-propagation neural networks implemented in MATLAB-Simulink environment. Four real power transformers working under different environment and climate conditions such as: desert, humid, cold were taken into account. The practical results of the diagnosis of these power transformers by the DGA are presented. Practical value.....
       


### [Pattern Inversion as a Pattern Recognition Method for Machine Learning](https://arxiv.org/abs/2108.10242)

**Authors:**
Alexei Mikhailov, Mikhail Karavay

**Abstract:**
Artificial neural networks use a lot of coefficients that take a great deal of computing power for their adjustment, especially if deep learning networks are employed. However, there exist coefficients-free extremely fast indexing-based technologies that work, for instance, in Google search engines, in genome sequencing, etc. The paper discusses the use of indexing-based methods for pattern recognition. It is shown that for pattern recognition applications such indexing methods replace with inverse patterns the fully inverted files, which are typically employed in search engines. Not only such inversion provide automatic feature extraction, which is a distinguishing mark of deep learning, but, unlike deep learning, pattern inversion supports almost instantaneous learning, which is a consequence of absence of coefficients. The paper discusses a pattern inversion formalism that makes use on a novel pattern transform and its application for unsupervised instant learning. Examples demonstrate a view-angle independent recognition of three-dimensional objects, such as cars, against arbitrary background, prediction of remaining useful life of aircraft engines, and other applications. In conclusion, it is noted that, in neurophysiology, the function of the neocortical mini-column has been widely debated since 1957. This paper hypothesize that, mathematically, the cortical mini-column can be described as an inverse pattern, which physically serves as a connection multiplier expanding associations of inputs with relevant pattern classes.
       


## September
### [An empirical evaluation of attention-based multi-head models for improved turbofan engine remaining useful life prediction](https://arxiv.org/abs/2109.01761)

**Authors:**
Abiodun Ayodeji, Wenhai Wang, Jianzhong Su, Jianquan Yuan, Xinggao Liu

**Abstract:**
A single unit (head) is the conventional input feature extractor in deep learning architectures trained on multivariate time series signals. The importance of the fixed-dimensional vector representation generated by the single-head network has been demonstrated for industrial machinery condition monitoring and predictive maintenance. However, processing heterogeneous sensor signals with a single-head may result in a model that cannot explicitly account for the diversity in time-varying multivariate inputs. This work extends the conventional single-head deep learning models to a more robust form by developing context-specific heads to independently capture the inherent pattern in each sensor reading. Using the turbofan aircraft engine benchmark dataset (CMAPSS), an extensive experiment is performed to verify the effectiveness and benefits of multi-head multilayer perceptron, recurrent networks, convolution network, the transformer-style stand-alone attention network, and their variants for remaining useful life estimation. Moreover, the effect of different attention mechanisms on the multi-head models is also evaluated. In addition, each architecture's relative advantage and computational overhead are analyzed. Results show that utilizing the attention layer is task-sensitive and model dependent, as it does not provide consistent improvement across the models investigated. The best model is further compared with five state-of-the-art models, and the comparison shows that a relatively simple multi-head architecture performs better than the state-of-the-art models. The results presented in this study demonstrate the importance of multi-head models and attention mechanisms to an improved understanding of the remaining useful life of industrial assets.
       


### [Remaining Useful Life Estimation of Hard Disk Drives using Bidirectional LSTM Networks](https://arxiv.org/abs/2109.05351)

**Authors:**
Austin Coursey, Gopal Nath, Srikanth Prabhu, Saptarshi Sengupta

**Abstract:**
Physical and cloud storage services are well-served by functioning and reliable high-volume storage systems. Recent observations point to hard disk reliability as one of the most pressing reliability issues in data centers containing massive volumes of storage devices such as HDDs. In this regard, early detection of impending failure at the disk level aids in reducing system downtime and reduces operational loss making proactive health monitoring a priority for AIOps in such settings. In this work, we introduce methods of extracting meaningful attributes associated with operational failure and of pre-processing the highly imbalanced health statistics data for subsequent prediction tasks using data-driven approaches. We use a Bidirectional LSTM with a multi-day look back period to learn the temporal progression of health indicators and baseline them against vanilla LSTM and Random Forest models to come up with several key metrics that establish the usefulness of and superiority of our model under some tightly defined operational constraints. For example, using a 15 day look back period, our approach can predict the occurrence of disk failure with an accuracy of 96.4% considering test data 60 days before failure. This helps to alert operations maintenance well in-advance about potential mitigation needs. In addition, our model reports a mean absolute error of 0.12 for predicting failure up to 60 days in advance, placing it among the state-of-the-art in recent literature.
       


### [The Monitoring, Logging, and Alarm system for the Cherenkov Telescope Array](https://arxiv.org/abs/2109.05770)

**Authors:**
Alessandro Costa, Kevin Munari, Federico Incardona, Pietro Bruno, Stefano Germani, Alessandro Grillo, Igor Oya, Eva Sciacca, Ugo Becciani, Mario Raciti

**Abstract:**
We present the current development of the Monitoring, Logging and Alarm subsystems in the framework of the Array Control and Data Acquisition System (ACADA) for the Cherenkov Telescope Array (CTA). The Monitoring System (MON) is the subsystem responsible for monitoring and logging the overall array (at each of the CTA sites) through the acquisition of monitoring and logging information from the array elements. The MON allows us to perform a systematic approach to fault detection and diagnosis supporting corrective and predictive maintenance to minimize the downtime of the system. We present a unified tool for monitoring data items from the telescopes and other devices deployed at the CTA array sites. Data are immediately available for the operator interface and quick-look quality checks and stored for later detailed inspection. The Array Alarm System (AAS) is the subsystem that provides the service that gathers, filters, exposes, and persists alarms raised by both the ACADA processes and the array elements supervised by the ACADA system. It collects alarms from the telescopes, the array calibration, the environmental monitoring instruments and the ACADA systems. The AAS sub-system also creates new alarms based on the analysis and correlation of the system software logs and the status of the system hardware providing the filter mechanisms for all the alarms. Data from the alarm system are then sent to the operator via the human-machine interface.
       


### [The state of health of the Russian population during the pandemic (according to sample surveys)](https://arxiv.org/abs/2109.05917)

**Authors:**
Leysan Anvarovna Davletshina, Natalia Alekseevna Sadovnikova, Alexander Valeryevich Bezrukov, Olga Guryevna Lebedinskaya

**Abstract:**
The article analyzes the population's assessment of their own health and attitude to a healthy lifestyle in the context of distribution by age groups. Of particular interest is the presence of transformations taking into account the complex epidemiological situation, the increase in the incidence of coronavirus infection in the population (the peak of the incidence came during the period of selective observation in 2020). The article assesses the closeness of the relationship between the respondents ' belonging to a particular socio-demographic group and their social well-being during the period of self-isolation, quarantine or other restrictions imposed during the coronavirus pandemic in 2020. To solve this problem, the demographic and socio-economic characteristics of respondents are presented, the distribution of responses according to the survey results is estimated and the most significant factor characteristics are selected. The distributions of respondents ' responses are presented for the selected questions. To determine the closeness of the relationship between the respondents ' answers to the question and their gender or age distribution, the coefficients of mutual conjugacy and rank correlation coefficients were calculated and analyzed. The ultimate goal of the analytical component of this study is to determine the social well-being of the Russian population during the pandemic on the basis of sample survey data. As a result of the analysis of changes for the period 2019-2020, the assessment of the closeness of communication revealed the parameters that form differences (gender, wealth, territory of residence).
       


### [Universal Adversarial Attack on Deep Learning Based Prognostics](https://arxiv.org/abs/2109.07142)

**Authors:**
Arghya Basak, Pradeep Rathore, Sri Harsha Nistala, Sagar Srinivas, Venkataramana Runkana

**Abstract:**
Deep learning-based time series models are being extensively utilized in engineering and manufacturing industries for process control and optimization, asset monitoring, diagnostic and predictive maintenance. These models have shown great improvement in the prediction of the remaining useful life (RUL) of industrial equipment but suffer from inherent vulnerability to adversarial attacks. These attacks can be easily exploited and can lead to catastrophic failure of critical industrial equipment. In general, different adversarial perturbations are computed for each instance of the input data. This is, however, difficult for the attacker to achieve in real time due to higher computational requirement and lack of uninterrupted access to the input data. Hence, we present the concept of universal adversarial perturbation, a special imperceptible noise to fool regression based RUL prediction models. Attackers can easily utilize universal adversarial perturbations for real-time attack since continuous access to input data and repetitive computation of adversarial perturbations are not a prerequisite for the same. We evaluate the effect of universal adversarial attacks using NASA turbofan engine dataset. We show that addition of universal adversarial perturbation to any instance of the input data increases error in the output predicted by the model. To the best of our knowledge, we are the first to study the effect of the universal adversarial perturbation on time series regression models. We further demonstrate the effect of varying the strength of perturbations on RUL prediction models and found that model accuracy decreases with the increase in perturbation strength of the universal adversarial attack. We also showcase that universal adversarial perturbation can be transferred across different models.
       


### [Learning to Rank Anomalies: Scalar Performance Criteria and Maximization of Two-Sample Rank Statistics](https://arxiv.org/abs/2109.09590)

**Authors:**
Myrto Limnios, Nathan Noiry, Stéphan Clémençon

**Abstract:**
The ability to collect and store ever more massive databases has been accompanied by the need to process them efficiently.  In many cases, most observations have the same behavior, while a probable small proportion of these observations are abnormal. Detecting the latter, defined as outliers, is one of the major challenges for machine learning applications (e.g. in fraud detection or in predictive maintenance). In this paper, we propose a methodology addressing the problem of outlier detection, by learning a data-driven scoring function defined on the feature space which reflects the degree of abnormality of the observations. This scoring function is learnt through a well-designed binary classification problem whose empirical criterion takes the form of a two-sample linear rank statistics on which theoretical results are available. We illustrate our methodology with preliminary encouraging numerical experiments.
       


### [Remaining useful life prediction with uncertainty quantification: development of a highly accurate model for rotating machinery](https://arxiv.org/abs/2109.11579)

**Authors:**
Zhaoyi Xu, Yanjie Guo, Joseph Homer Saleh

**Abstract:**
Rotating machinery is essential to modern life, from power generation to transportation and a host of other industrial applications. Since such equipment generally operates under challenging working conditions, which can lead to untimely failures, accurate remaining useful life (RUL) prediction is essential for maintenance planning and to prevent catastrophic failures. In this work, we address current challenges in data-driven RUL prediction for rotating machinery. The challenges revolve around the accuracy and uncertainty quantification of the prediction, and the non-stationarity of the system degradation and RUL estimation given sensor data. We devise a novel architecture and RUL prediction model with uncertainty quantification, termed VisPro, which integrates time-frequency analysis, deep learning image recognition, and nonstationary Gaussian process regression. We analyze and benchmark the results obtained with our model against those of other advanced data-driven RUL prediction models for rotating machinery using the PHM12 bearing vibration dataset. The computational experiments show that (1) the VisPro predictions are highly accurate and provide significant improvements over existing prediction models (three times more accurate than the second-best model), and (2) the RUL uncertainty bounds are valid and informative. We identify and discuss the architectural and modeling choices made that explain this excellent predictive performance of VisPro.
       


### [Predicting pigging operations in oil pipelines](https://arxiv.org/abs/2109.11812)

**Authors:**
Riccardo Angelo Giro, Giancarlo Bernasconi, Giuseppe Giunta, Simone Cesari

**Abstract:**
This paper presents an innovative machine learning methodology that leverages on long-term vibroacoustic measurements to perform automated predictions of the needed pigging operations in crude oil trunklines. Historical pressure signals have been collected by Eni (e-vpms monitoring system) for two years on discrete points at a relative distance of 30-35 km along an oil pipeline (100 km length, 16 inch diameter pipes) located in Northern Italy. In order to speed up the activity and to check the operation logs, a tool has been implemented to automatically highlight the historical pig operations performed on the line. Such a tool is capable of detecting, in the observed pressure measurements, the acoustic noise generated by the travelling pig. All the data sets have been reanalyzed and exploited by using field data validations to guide a decision tree regressor (DTR). Several statistical indicators, computed from pressure head loss between line segments, are fed to the DTR, which automatically outputs probability values indicating the possible need for pigging the pipeline. The procedure is applied to the vibroacoustic signals of each pair of consecutive monitoring stations, such that the proposed predictive maintenance strategy is capable of tracking the conditions of individual pipeline sections, thus determining which portion of the conduit is subject to the highest occlusion levels in order to optimize the clean-up operations. Prediction accuracy is assessed by evaluating the typical metrics used in statistical analysis of regression problems, such as the Root Mean Squared Error (RMSE).
       


### [Accurate Remaining Useful Life Prediction with Uncertainty Quantification: a Deep Learning and Nonstationary Gaussian Process Approach](https://arxiv.org/abs/2109.12111)

**Authors:**
Zhaoyi Xu, Yanjie Guo, Joseph Homer Saleh

**Abstract:**
Remaining useful life (RUL) refers to the expected remaining lifespan of a component or system. Accurate RUL prediction is critical for prognostic and health management and for maintenance planning. In this work, we address three prevalent challenges in data-driven RUL prediction, namely the handling of high dimensional input features, the robustness to noise in sensor data and prognostic datasets, and the capturing of the time-dependency between system degradation and RUL prediction. We devise a highly accurate RUL prediction model with uncertainty quantification, which integrates and leverages the advantages of deep learning and nonstationary Gaussian process regression (DL-NSGPR). We examine and benchmark our model against other advanced data-driven RUL prediction models using the turbofan engine dataset from the NASA prognostic repository. Our computational experiments show that the DL-NSGPR predictions are highly accurate with root mean square error 1.7 to 6.2 times smaller than those of competing RUL models. Furthermore, the results demonstrate that RUL uncertainty bounds with the proposed DL-NSGPR are both valid and significantly tighter than other stochastic RUL prediction models. We unpack and discuss the reasons for this excellent performance of the DL-NSGPR.
       


### [Generalized multiscale feature extraction for remaining useful life prediction of bearings with generative adversarial networks](https://arxiv.org/abs/2109.12513)

**Authors:**
Sungho Suh, Paul Lukowicz, Yong Oh Lee

**Abstract:**
Bearing is a key component in industrial machinery and its failure may lead to unwanted downtime and economic loss. Hence, it is necessary to predict the remaining useful life (RUL) of bearings. Conventional data-driven approaches of RUL prediction require expert domain knowledge for manual feature extraction and may suffer from data distribution discrepancy between training and test data. In this study, we propose a novel generalized multiscale feature extraction method with generative adversarial networks. The adversarial training learns the distribution of training data from different bearings and is introduced for health stage division and RUL prediction. To capture the sequence feature from a one-dimensional vibration signal, we adapt a U-Net architecture that reconstructs features to process them with multiscale layers in the generator of the adversarial network. To validate the proposed method, comprehensive experiments on two rotating machinery datasets have been conducted to predict the RUL. The experimental results show that the proposed feature extraction method can effectively predict the RUL and outperforms the conventional RUL prediction approaches based on deep neural networks. The implementation code is available at https://github.com/opensuh/GMFE.
       


### [Lithium-ion Battery State of Health Estimation based on Cycle Synchronization using Dynamic Time Warping](https://arxiv.org/abs/2109.13448)

**Authors:**
Kate Qi Zhou, Yan Qin, Billy Pik Lik Lau, Chau Yuen, Stefan Adams

**Abstract:**
The state of health (SOH) estimation plays an essential role in battery-powered applications to avoid unexpected breakdowns due to battery capacity fading. However, few studies have paid attention to the problem of uneven length of degrading cycles, simply employing manual operation or leaving to the automatic processing mechanism of advanced machine learning models, like long short-term memory (LSTM). As a result, this causes information loss and caps the full capability of the data-driven SOH estimation models. To address this challenge, this paper proposes an innovative cycle synchronization way to change the existing coordinate system using dynamic time warping, not only enabling the equal length inputs of the estimation model but also preserving all information. By exploiting the time information of the time series, the proposed method embeds the time index and the original measurements into a novel indicator to reflect the battery degradation status, which could have the same length over cycles. Adopting the LSTM as the basic estimation model, the cycle synchronization-based SOH model could significantly improve the prediction accuracy by more than 30% compared to the traditional LSTM.
       


### [An Offline Deep Reinforcement Learning for Maintenance Decision-Making](https://arxiv.org/abs/2109.15050)

**Authors:**
Hamed Khorasgani, Haiyan Wang, Chetan Gupta, Ahmed Farahat

**Abstract:**
Several machine learning and deep learning frameworks have been proposed to solve remaining useful life estimation and failure prediction problems in recent years. Having access to the remaining useful life estimation or likelihood of failure in near future helps operators to assess the operating conditions and, therefore, provides better opportunities for sound repair and maintenance decisions. However, many operators believe remaining useful life estimation and failure prediction solutions are incomplete answers to the maintenance challenge. They argue that knowing the likelihood of failure in the future is not enough to make maintenance decisions that minimize costs and keep the operators safe. In this paper, we present a maintenance framework based on offline supervised deep reinforcement learning that instead of providing information such as likelihood of failure, suggests actions such as "continuation of the operation" or "the visitation of the repair shop" to the operators in order to maximize the overall profit. Using offline reinforcement learning makes it possible to learn the optimum maintenance policy from historical data without relying on expensive simulators. We demonstrate the application of our solution in a case study using the NASA C-MAPSS dataset.
       


## October
### [Real-Time Predictive Maintenance using Autoencoder Reconstruction and Anomaly Detection](https://arxiv.org/abs/2110.01447)

**Authors:**
Sean Givnan, Carl Chalmers, Paul Fergus, Sandra Ortega, Tom Whalley

**Abstract:**
Rotary machine breakdown detection systems are outdated and dependent upon routine testing to discover faults. This is costly and often reactive in nature. Real-time monitoring offers a solution for detecting faults without the need for manual observation. However, manual interpretation for threshold anomaly detection is often subjective and varies between industrial experts. This approach is ridged and prone to a large number of false positives. To address this issue, we propose a Machine Learning (ML) approach to model normal working operation and detect anomalies. The approach extracts key features from signals representing known normal operation to model machine behaviour and automatically identify anomalies. The ML learns generalisations and generates thresholds based on fault severity. This provides engineers with a traffic light system were green is normal behaviour, amber is worrying and red signifies a machine fault. This scale allows engineers to undertake early intervention measures at the appropriate time. The approach is evaluated on windowed real machine sensor data to observe normal and abnormal behaviour. The results demonstrate that it is possible to detect anomalies within the amber range and raise alarms before machine failure.
       


### [To Charge or to Sell? EV Pack Useful Life Estimation via LSTMs, CNNs, and Autoencoders](https://arxiv.org/abs/2110.03585)

**Authors:**
Michael Bosello, Carlo Falcomer, Claudio Rossi, Giovanni Pau

**Abstract:**
Electric vehicles (EVs) are spreading fast as they promise to provide better performance and comfort, but above all, to help face climate change. Despite their success, their cost is still a challenge. Lithium-ion batteries are one of the most expensive EV components, and have become the standard for energy storage in various applications. Precisely estimating the remaining useful life (RUL) of battery packs can encourage their reuse and thus help to reduce the cost of EVs and improve sustainability. A correct RUL estimation can be used to quantify the residual market value of the battery pack. The customer can then decide to sell the battery when it still has a value, i.e., before it exceeds the end of life of the target application, so it can still be reused in a second domain without compromising safety and reliability. This paper proposes and compares two deep learning approaches to estimate the RUL of Li-ion batteries: LSTM and autoencoders vs. CNN and autoencoders. The autoencoders are used to extract useful features, while the subsequent network is then used to estimate the RUL. Compared to what has been proposed so far in the literature, we employ measures to ensure the method's applicability in the actual deployed application. Such measures include (1) avoiding using non-measurable variables as input, (2) employing appropriate datasets with wide variability and different conditions, and (3) predicting the remaining ampere-hours instead of the number of cycles. The results show that the proposed methods can generalize on datasets consisting of numerous batteries with high variance.
       


### [Predictive Maintenance for General Aviation Using Convolutional Transformers](https://arxiv.org/abs/2110.03757)

**Authors:**
Hong Yang, Aidan LaBella, Travis Desell

**Abstract:**
Predictive maintenance systems have the potential to significantly reduce costs for maintaining aircraft fleets as well as provide improved safety by detecting maintenance issues before they come severe. However, the development of such systems has been limited due to a lack of publicly labeled multivariate time series (MTS) sensor data. MTS classification has advanced greatly over the past decade, but there is a lack of sufficiently challenging benchmarks for new methods. This work introduces the NGAFID Maintenance Classification (NGAFID-MC) dataset as a novel benchmark in terms of difficulty, number of samples, and sequence length. NGAFID-MC consists of over 7,500 labeled flights, representing over 11,500 hours of per second flight data recorder readings of 23 sensor parameters. Using this benchmark, we demonstrate that Recurrent Neural Network (RNN) methods are not well suited for capturing temporally distant relationships and propose a new architecture called Convolutional Multiheaded Self Attention (Conv-MHSA) that achieves greater classification performance at greater computational efficiency. We also demonstrate that image inspired augmentations of cutout, mixup, and cutmix, can be used to reduce overfitting and improve generalization in MTS classification. Our best trained models have been incorporated back into the NGAFID to allow users to potentially detect flights that require maintenance as well as provide feedback to further expand and refine the NGAFID-MC dataset.
       


### [A Survey on Proactive Customer Care: Enabling Science and Steps to Realize it](https://arxiv.org/abs/2110.05015)

**Authors:**
Viswanath Ganapathy, Sauptik Dhar, Olimpiya Saha, Pelin Kurt Garberson, Javad Heydari, Mohak Shah

**Abstract:**
In recent times, advances in artificial intelligence (AI) and IoT have enabled seamless and viable maintenance of appliances in home and building environments. Several studies have shown that AI has the potential to provide personalized customer support which could predict and avoid errors more reliably than ever before. In this paper, we have analyzed the various building blocks needed to enable a successful AI-driven predictive maintenance use-case. Unlike, existing surveys which mostly provide a deep dive into the recent AI algorithms for Predictive Maintenance (PdM), our survey provides the complete view; starting from business impact to recent technology advancements in algorithms as well as systems research and model deployment. Furthermore, we provide exemplar use-cases on predictive maintenance of appliances using publicly available data sets. Our survey can serve as a template needed to design a successful predictive maintenance use-case. Finally, we touch upon existing public data sources and provide a step-wise breakdown of an AI-driven proactive customer care (PCC) use-case, starting from generic anomaly detection to fault prediction and finally root-cause analysis. We highlight how such a step-wise approach can be advantageous for accurate model building and helpful for gaining insights into predictive maintenance of electromechanical appliances.
       


### [Remote Anomaly Detection in Industry 4.0 Using Resource-Constrained Devices](https://arxiv.org/abs/2110.05757)

**Authors:**
Anders E. Kalør, Daniel Michelsanti, Federico Chiariotti, Zheng-Hua Tan, Petar Popovski

**Abstract:**
A central use case for the Internet of Things (IoT) is the adoption of sensors to monitor physical processes, such as the environment and industrial manufacturing processes, where they provide data for predictive maintenance, anomaly detection, or similar. The sensor devices are typically resource-constrained in terms of computation and power, and need to rely on cloud or edge computing for data processing. However, the capacity of the wireless link and their power constraints limit the amount of data that can be transmitted to the cloud. While this is not problematic for the monitoring of slowly varying processes such as temperature, it is more problematic for complex signals such as those captured by vibration and acoustic sensors. In this paper, we consider the specific problem of remote anomaly detection based on signals that fall into the latter category over wireless channels with resource-constrained sensors. We study the impact of source coding on the detection accuracy with both an anomaly detector based on Principal Component Analysis (PCA) and one based on an autoencoder. We show that the coded transmission is beneficial when the signal-to-noise ratio (SNR) of the channel is low, while uncoded transmission performs best in the high SNR regime.
       


### [StreaMulT: Streaming Multimodal Transformer for Heterogeneous and Arbitrary Long Sequential Data](https://arxiv.org/abs/2110.08021)

**Authors:**
Victor Pellegrain, Myriam Tami, Michel Batteux, Céline Hudelot

**Abstract:**
The increasing complexity of Industry 4.0 systems brings new challenges regarding predictive maintenance tasks such as fault detection and diagnosis. A corresponding and realistic setting includes multi-source data streams from different modalities, such as sensors measurements time series, machine images, textual maintenance reports, etc. These heterogeneous multimodal streams also differ in their acquisition frequency, may embed temporally unaligned information and can be arbitrarily long, depending on the considered system and task. Whereas multimodal fusion has been largely studied in a static setting, to the best of our knowledge, there exists no previous work considering arbitrarily long multimodal streams alongside with related tasks such as prediction across time. Thus, in this paper, we first formalize this paradigm of heterogeneous multimodal learning in a streaming setting as a new one. To tackle this challenge, we propose StreaMulT, a Streaming Multimodal Transformer relying on cross-modal attention and on a memory bank to process arbitrarily long input sequences at training time and run in a streaming way at inference. StreaMulT improves the state-of-the-art metrics on CMU-MOSEI dataset for Multimodal Sentiment Analysis task, while being able to deal with much longer inputs than other multimodal models. The conducted experiments eventually highlight the importance of the textual embedding layer, questioning recent improvements in Multimodal Sentiment Analysis benchmarks.
       


### [Towards modelling hazard factors in unstructured data spaces using gradient-based latent interpolation](https://arxiv.org/abs/2110.11312)

**Authors:**
Tobias Weber, Michael Ingrisch, Bernd Bischl, David Rügamer

**Abstract:**
The application of deep learning in survival analysis (SA) allows utilizing unstructured and high-dimensional data types uncommon in traditional survival methods. This allows to advance methods in fields such as digital health, predictive maintenance, and churn analysis, but often yields less interpretable and intuitively understandable models due to the black-box character of deep learning-based approaches. We close this gap by proposing 1) a multi-task variational autoencoder (VAE) with survival objective, yielding survival-oriented embeddings, and 2) a novel method HazardWalk that allows to model hazard factors in the original data space. HazardWalk transforms the latent distribution of our autoencoder into areas of maximized/minimized hazard and then uses the decoder to project changes to the original domain. Our procedure is evaluated on a simulated dataset as well as on a dataset of CT imaging data of patients with liver metastases.
       


### [Noninvasive ultrasound for Lithium-ion batteries state estimation](https://arxiv.org/abs/2110.14033)

**Authors:**
Simon Montoya-Bedoya, Miguel Bernal, Laura A. Sabogal-Moncada, Hader V. Martinez-Tejada, Esteban Garcia-Tamayo

**Abstract:**
Lithium-ion battery degradation estimation using fast and noninvasive techniques is a crucial issue in the circular economy framework of this technology. Currently, most of the approaches used to establish the battery-state (i.e., State of Charge (SoC), State of Health (SoH)) require time-consuming processes. In the present preliminary study, an ultrasound array was used to assess the influence of the SoC and SoH on the variations in the time of flight (TOF) and the speed of sound (SOS) of the ultrasound wave inside the batteries. Nine aged 18650 Lithium-ion batteries were imaged at 100% and 0% SoC using a Vantage-256 system (Verasonics, Inc.) equipped with a 64-element ultrasound array and a center frequency of 5 MHz (Imasonic SAS). It was found that second-life batteries have a complex ultrasound response due to the presence of many degradation pathways and, thus, making it harder to analyze the ultrasound measurements. Although further analysis must be done to elucidate a clear correlation between changes in the ultrasound wave properties and the battery state estimation, this approach seems very promising for future nondestructive evaluation of second-life batteries.
       


## November
### [A stacked deep convolutional neural network to predict the remaining useful life of a turbofan engine](https://arxiv.org/abs/2111.12689)

**Authors:**
David Solis-Martin, Juan Galan-Paez, Joaquin Borrego-Diaz

**Abstract:**
This paper presents the data-driven techniques and methodologies used to predict the remaining useful life (RUL) of a fleet of aircraft engines that can suffer failures of diverse nature. The solution presented is based on two Deep Convolutional Neural Networks (DCNN) stacked in two levels. The first DCNN is used to extract a low-dimensional feature vector using the normalized raw data as input. The second DCNN ingests a list of vectors taken from the former DCNN and estimates the RUL. Model selection was carried out by means of Bayesian optimization using a repeated random subsampling validation approach. The proposed methodology was ranked in the third place of the 2021 PHM Conference Data Challenge.
       


### [Overcoming limited battery data challenges: A coupled neural network approach](https://arxiv.org/abs/2111.15348)

**Authors:**
Aniruddh Herle, Janamejaya Channegowda, Dinakar Prabhu

**Abstract:**
The Electric Vehicle (EV) Industry has seen extraordinary growth in the last few years. This is primarily due to an ever increasing awareness of the detrimental environmental effects of fossil fuel powered vehicles and availability of inexpensive Lithium-ion batteries (LIBs). In order to safely deploy these LIBs in Electric Vehicles, certain battery states need to be constantly monitored to ensure safe and healthy operation. The use of Machine Learning to estimate battery states such as State-of-Charge and State-of-Health have become an extremely active area of research. However, limited availability of open-source diverse datasets has stifled the growth of this field, and is a problem largely ignored in literature. In this work, we propose a novel method of time-series battery data augmentation using deep neural networks. We introduce and analyze the method of using two neural networks working together to alternatively produce synthetic charging and discharging battery profiles. One model produces battery charging profiles, and another produces battery discharging profiles. The proposed approach is evaluated using few public battery datasets to illustrate its effectiveness, and our results show the efficacy of this approach to solve the challenges of limited battery data. We also test this approach on dynamic Electric Vehicle drive cycles as well.
       


## December
### [Semi-Supervised Surface Anomaly Detection of Composite Wind Turbine Blades From Drone Imagery](https://arxiv.org/abs/2112.00556)

**Authors:**
Jack. W. Barker, Neelanjan Bhowmik, Toby. P. Breckon

**Abstract:**
Within commercial wind energy generation, the monitoring and predictive maintenance of wind turbine blades in-situ is a crucial task, for which remote monitoring via aerial survey from an Unmanned Aerial Vehicle (UAV) is commonplace. Turbine blades are susceptible to both operational and weather-based damage over time, reducing the energy efficiency output of turbines. In this study, we address automating the otherwise time-consuming task of both blade detection and extraction, together with fault detection within UAV-captured turbine blade inspection imagery. We propose BladeNet, an application-based, robust dual architecture to perform both unsupervised turbine blade detection and extraction, followed by super-pixel generation using the Simple Linear Iterative Clustering (SLIC) method to produce regional clusters. These clusters are then processed by a suite of semi-supervised detection methods. Our dual architecture detects surface faults of glass fibre composite material blades with high aptitude while requiring minimal prior manual image annotation. BladeNet produces an Average Precision (AP) of 0.995 across our Ørsted blade inspection dataset for offshore wind turbines and 0.223 across the Danish Technical University (DTU) NordTank turbine blade inspection dataset. BladeNet also obtains an AUC of 0.639 for surface anomaly detection across the Ørsted blade inspection dataset.
       


### [A recurrent neural network approach for remaining useful life prediction utilizing a novel trend features construction method](https://arxiv.org/abs/2112.05372)

**Authors:**
Sen Zhao, Yong Zhang, Shang Wang, Beitong Zhou, Cheng Cheng

**Abstract:**
Data-driven methods for remaining useful life (RUL) prediction normally learn features from a fixed window size of a priori of degradation, which may lead to less accurate prediction results on different datasets because of the variance of local features. This paper proposes a method for RUL prediction which depends on a trend feature representing the overall time sequence of degradation. Complete ensemble empirical mode decomposition, followed by a reconstruction procedure, is created to build the trend features. The probability distribution of sensors' measurement learned by conditional neural processes is used to evaluate the trend features. With the best trend feature, a data-driven model using long short-term memory is developed to predict the RUL. To prove the effectiveness of the proposed method, experiments on a benchmark C-MAPSS dataset are carried out and compared with other state-of-the-art methods. Comparison results show that the proposed method achieves the smallest root mean square values in prediction of all RUL.
       


### [MTV: Visual Analytics for Detecting, Investigating, and Annotating Anomalies in Multivariate Time Series](https://arxiv.org/abs/2112.05734)

**Authors:**
Dongyu Liu, Sarah Alnegheimish, Alexandra Zytek, Kalyan Veeramachaneni

**Abstract:**
Detecting anomalies in time-varying multivariate data is crucial in various industries for the predictive maintenance of equipment. Numerous machine learning (ML) algorithms have been proposed to support automated anomaly identification. However, a significant amount of human knowledge is still required to interpret, analyze, and calibrate the results of automated analysis. This paper investigates current practices used to detect and investigate anomalies in time series data in industrial contexts and identifies corresponding needs. Through iterative design and working with nine experts from two industry domains (aerospace and energy), we characterize six design elements required for a successful visualization system that supports effective detection, investigation, and annotation of time series anomalies. We summarize an ideal human-AI collaboration workflow that streamlines the process and supports efficient and collaborative analysis. We introduce MTV (Multivariate Time Series Visualization), a visual analytics system to support such workflow. The system incorporates a set of novel visualization and interaction designs to support multi-faceted time series exploration, efficient in-situ anomaly annotation, and insight communication. Two user studies, one with 6 spacecraft experts (with routine anomaly analysis tasks) and one with 25 general end-users (without such tasks), are conducted to demonstrate the effectiveness and usefulness of MTV.
       


### [Practical Quantum K-Means Clustering: Performance Analysis and Applications in Energy Grid Classification](https://arxiv.org/abs/2112.08506)

**Authors:**
Stephen DiAdamo, Corey O'Meara, Giorgio Cortiana, Juan Bernabé-Moreno

**Abstract:**
In this work, we aim to solve a practical use-case of unsupervised clustering which has applications in predictive maintenance in the energy operations sector using quantum computers. Using only cloud access to quantum computers, we complete a thorough performance analysis of what some current quantum computing systems are capable of for practical applications involving non-trivial mid-to-high dimensional datasets. We first benchmark how well distance estimation can be performed using two different metrics based on the swap-test, using angle and amplitude data embedding. Next, for the clustering performance analysis, we generate sets of synthetic data with varying cluster variance and compare simulation to physical hardware results using the two metrics. From the results of this performance analysis, we propose a general, competitive, and parallelized version of quantum $k$-means clustering to avoid some pitfalls discovered due to noisy hardware and apply the approach to a real energy grid clustering scenario. Using real-world German electricity grid data, we show that the new approach improves the balanced accuracy of the standard quantum $k$-means clustering by $67.8\%$ with respect to the labeling of the classical algorithm.
       


### [Hybrid Classical-Quantum Autoencoder for Anomaly Detection](https://arxiv.org/abs/2112.08869)

**Authors:**
Alona Sakhnenko, Corey O'Meara, Kumar J. B. Ghosh, Christian B. Mendl, Giorgio Cortiana, Juan Bernabé-Moreno

**Abstract:**
We propose a Hybrid classical-quantum Autoencoder (HAE) model, which is a synergy of a classical autoencoder (AE) and a parametrized quantum circuit (PQC) that is inserted into its bottleneck. The PQC augments the latent space, on which a standard outlier detection method is applied to search for anomalous data points within a classical dataset. Using this model and applying it to both standard benchmarking datasets, and a specific use-case dataset which relates to predictive maintenance of gas power plants, we show that the addition of the PQC leads to a performance enhancement in terms of precision, recall, and F1 score. Furthermore, we probe different PQC Ansätze and analyse which PQC features make them effective for this task.
       


### [A deep reinforcement learning model for predictive maintenance planning of road assets: Integrating LCA and LCCA](https://arxiv.org/abs/2112.12589)

**Authors:**
Moein Latifi, Fateme Golivand Darvishvand, Omid Khandel, Mobin Latifi Nowsoud

**Abstract:**
Road maintenance planning is an integral part of road asset management. One of the main challenges in Maintenance and Rehabilitation (M&R) practices is to determine maintenance type and timing. This research proposes a framework using Reinforcement Learning (RL) based on the Long Term Pavement Performance (LTPP) database to determine the type and timing of M&R practices. A predictive DNN model is first developed in the proposed algorithm, which serves as the Environment for the RL algorithm. For the Policy estimation of the RL model, both DQN and PPO models are developed. However, PPO has been selected in the end due to better convergence and higher sample efficiency. Indicators used in this study are International Roughness Index (IRI) and Rutting Depth (RD). Initially, we considered Cracking Metric (CM) as the third indicator, but it was then excluded due to the much fewer data compared to other indicators, which resulted in lower accuracy of the results. Furthermore, in cost-effectiveness calculation (reward), we considered both the economic and environmental impacts of M&R treatments. Costs and environmental impacts have been evaluated with paLATE 2.0 software. Our method is tested on a hypothetical case study of a six-lane highway with 23 kilometers length located in Texas, which has a warm and wet climate. The results propose a 20-year M&R plan in which road condition remains in an excellent condition range. Because the early state of the road is at a good level of service, there is no need for heavy maintenance practices in the first years. Later, after heavy M&R actions, there are several 1-2 years of no need for treatments. All of these show that the proposed plan has a logical result. Decision-makers and transportation agencies can use this scheme to conduct better maintenance practices that can prevent budget waste and, at the same time, minimize the environmental impacts.
       


### [A general framework for penalized mixed-effects multitask learning with applications on DNA methylation surrogate biomarkers creation](https://arxiv.org/abs/2112.12719)

**Authors:**
Andrea Cappozzo, Francesca Ieva, Giovanni Fiorito

**Abstract:**
Recent evidence highlights the usefulness of DNA methylation (DNAm) biomarkers as surrogates for exposure to risk factors for non-communicable diseases in epidemiological studies and randomized trials. DNAm variability has been demonstrated to be tightly related to lifestyle behavior and exposure to environmental risk factors, ultimately providing an unbiased proxy of an individual state of health. At present, the creation of DNAm surrogates relies on univariate penalized regression models, with elastic-net regularizer being the gold standard when accomplishing the task. Nonetheless, more advanced modeling procedures are required in the presence of multivariate outcomes with a structured dependence pattern among the study samples. In this work we propose a general framework for mixed-effects multitask learning in presence of high-dimensional predictors to develop a multivariate DNAm biomarker from a multi-center study. A penalized estimation scheme based on an expectation-maximization algorithm is devised, in which any penalty criteria for fixed-effects models can be conveniently incorporated in the fitting process. We apply the proposed methodology to create novel DNAm surrogate biomarkers for multiple correlated risk factors for cardiovascular diseases and comorbidities. We show that the proposed approach, modeling multiple outcomes together, outperforms state-of-the-art alternatives, both in predictive power and bio-molecular interpretation of the results.
       


### [Integrating Physics-Based Modeling with Machine Learning for Lithium-Ion Batteries](https://arxiv.org/abs/2112.12979)

**Authors:**
Hao Tu, Scott Moura, Yebin Wang, Huazhen Fang

**Abstract:**
Mathematical modeling of lithium-ion batteries (LiBs) is a primary challenge in advanced battery management. This paper proposes two new frameworks to integrate physics-based models with machine learning to achieve high-precision modeling for LiBs. The frameworks are characterized by informing the machine learning model of the state information of the physical model, enabling a deep integration between physics and machine learning. Based on the frameworks, a series of hybrid models are constructed, through combining an electrochemical model and an equivalent circuit model, respectively, with a feedforward neural network. The hybrid models are relatively parsimonious in structure and can provide considerable voltage predictive accuracy under a broad range of C-rates, as shown by extensive simulations and experiments. The study further expands to conduct aging-aware hybrid modeling, leading to the design of a hybrid model conscious of the state-of-health to make prediction. The experiments show that the model has high voltage predictive accuracy throughout a LiB's cycle life.
       


### [Predicting Breakdown Risk Based on Historical Maintenance Data for Air Force Ground Vehicles](https://arxiv.org/abs/2112.13922)

**Authors:**
Jeff Jang, Dilan Nana, Jack Hochschild, Jordi Vila Hernandez de Lorenzo

**Abstract:**
Unscheduled maintenance has contributed to longer downtime for vehicles and increased costs for Logistic Readiness Squadrons (LRSs) in the Air Force. When vehicles are in need of repair outside of their scheduled time, depending on their priority level, the entire squadron's slated repair schedule is transformed negatively. The repercussions of unscheduled maintenance are specifically seen in the increase of man hours required to maintain vehicles that should have been working well: this can include more man hours spent on maintenance itself, waiting for parts to arrive, hours spent re-organizing the repair schedule, and more. The dominant trend in the current maintenance system at LRSs is that they do not have predictive maintenance infrastructure to counteract the influx of unscheduled repairs they experience currently, and as a result, their readiness and performance levels are lower than desired.
  We use data pulled from the Defense Property and Accountability System (DPAS), that the LRSs currently use to store their vehicle maintenance information. Using historical vehicle maintenance data we receive from DPAS, we apply three different algorithms independently to construct an accurate predictive system to optimize maintenance schedules at any given time. Through the application of Logistics Regression, Random Forest, and Gradient Boosted Trees algorithms, we found that a Logistic Regression algorithm, fitted to our data, produced the most accurate results. Our findings indicate that not only would continuing the use of Logistic Regression be prudent for our research purposes, but that there is opportunity to further tune and optimize our Logistic Regression model for higher accuracy.
       


# 2022
## January
### [Knowledge Informed Machine Learning using a Weibull-based Loss Function](https://arxiv.org/abs/2201.01769)

**Authors:**
Tim von Hahn, Chris K Mechefske

**Abstract:**
Machine learning can be enhanced through the integration of external knowledge. This method, called knowledge informed machine learning, is also applicable within the field of Prognostics and Health Management (PHM). In this paper, the various methods of knowledge informed machine learning, from a PHM context, are reviewed with the goal of helping the reader understand the domain. In addition, a knowledge informed machine learning technique is demonstrated, using the common IMS and PRONOSTIA bearing data sets, for remaining useful life (RUL) prediction. Specifically, knowledge is garnered from the field of reliability engineering which is represented through the Weibull distribution. The knowledge is then integrated into a neural network through a novel Weibull-based loss function. A thorough statistical analysis of the Weibull-based loss function is conducted, demonstrating the effectiveness of the method on the PRONOSTIA data set. However, the Weibull-based loss function is less effective on the IMS data set. The results, shortcomings, and benefits of the approach are discussed in length. Finally, all the code is publicly available for the benefit of other researchers.
       


### [Continuous-Time Probabilistic Models for Longitudinal Electronic Health Records](https://arxiv.org/abs/2201.03675)

**Authors:**
Alan D. Kaplan, Uttara Tipnis, Jean C. Beckham, Nathan A. Kimbrel, David W. Oslin, Benjamin H. McMahon

**Abstract:**
Analysis of longitudinal Electronic Health Record (EHR) data is an important goal for precision medicine. Difficulty in applying Machine Learning (ML) methods, either predictive or unsupervised, stems in part from the heterogeneity and irregular sampling of EHR data. We present an unsupervised probabilistic model that captures nonlinear relationships between variables over continuous-time. This method works with arbitrary sampling patterns and captures the joint probability distribution between variable measurements and the time intervals between them. Inference algorithms are derived that can be used to evaluate the likelihood of future using under a trained model. As an example, we consider data from the United States Veterans Health Administration (VHA) in the areas of diabetes and depression. Likelihood ratio maps are produced showing the likelihood of risk for moderate-severe vs minimal depression as measured by the Patient Health Questionnaire-9 (PHQ-9).
       


### [A Joint Chance-Constrained Stochastic Programming Approach for the Integrated Predictive Maintenance and Operations Scheduling Problem in Power Systems](https://arxiv.org/abs/2201.04178)

**Authors:**
Bahar Cennet Okumusoglu, Beste Basciftci, Burak Kocuk

**Abstract:**
Maintenance planning plays a key role in power system operations under uncertainty by helping system operators ensure a reliable and secure power grid. This paper studies a short-term condition-based integrated maintenance planning with operations scheduling problem while considering the unexpected failure possibilities of generators as well as transmission lines. We formulate this problem as a two-stage stochastic mixed-integer program with failure scenarios sampled from the sensor-driven remaining lifetime distributions of the individual system elements whereas a joint chance-constraint consisting of Poisson Binomial random variables is introduced to account for failure risks. Because of its intractability, we develop a cutting-plane method to obtain an exact reformulation of the joint chance-constraint by proposing a separation subroutine and deriving stronger cuts as part of this procedure. To solve large-scale instances, we derive a second-order cone programming based safe approximation of this constraint. Furthermore, we propose a decomposition-based algorithm implemented in parallel fashion for solving the resulting stochastic program, by exploiting the features of the integer L-shaped method and the special structure of the maintenance and operations scheduling problem to derive stronger optimality cuts. We further present preprocessing steps over transmission line flow constraints to identify redundancies. To illustrate the computational performance and efficiency of our algorithm compared to more conventional maintenance approaches, we design a computational study focusing on a weekly plan with daily maintenance and hourly operational decisions involving detailed unit commitment subproblems. Our computational results on various IEEE instances demonstrate the computational efficiency of the proposed approach with reliable and cost-effective maintenance and operational schedules.
       


### [Discrete Simulation Optimization for Tuning Machine Learning Method Hyperparameters](https://arxiv.org/abs/2201.05978)

**Authors:**
Varun Ramamohan, Shobhit Singhal, Aditya Raj Gupta, Nomesh Bhojkumar Bolia

**Abstract:**
Machine learning (ML) methods are used in most technical areas such as image recognition, product recommendation, financial analysis, medical diagnosis, and predictive maintenance. An important aspect of implementing ML methods involves controlling the learning process for the ML method so as to maximize the performance of the method under consideration. Hyperparameter tuning is the process of selecting a suitable set of ML method parameters that control its learning process. In this work, we demonstrate the use of discrete simulation optimization methods such as ranking and selection (R&S) and random search for identifying a hyperparameter set that maximizes the performance of a ML method. Specifically, we use the KN R&S method and the stochastic ruler random search method and one of its variations for this purpose. We also construct the theoretical basis for applying the KN method, which determines the optimal solution with a statistical guarantee via solution space enumeration. In comparison, the stochastic ruler method asymptotically converges to global optima and incurs smaller computational overheads. We demonstrate the application of these methods to a wide variety of machine learning models, including deep neural network models used for time series prediction and image classification. We benchmark our application of these methods with state-of-the-art hyperparameter optimization libraries such as $hyperopt$ and $mango$. The KN method consistently outperforms $hyperopt$'s random search (RS) and Tree of Parzen Estimators (TPE) methods. The stochastic ruler method outperforms the $hyperopt$ RS method and offers statistically comparable performance with respect to $hyperopt$'s TPE method and the $mango$ algorithm.
       


### [Blockchain based AI-enabled Industry 4.0 CPS Protection against Advanced Persistent Threat](https://arxiv.org/abs/2201.12727)

**Authors:**
Ziaur Rahman, Xun Yi Ibrahim Khalil

**Abstract:**
Industry 4.0 is all about doing things in a concurrent, secure, and fine-grained manner. IoT edge-sensors and their associated data play a predominant role in today's industry ecosystem. Breaching data or forging source devices after injecting advanced persistent threats (APT) damages the industry owners' money and loss of operators' lives. The existing challenges include APT injection attacks targeting vulnerable edge devices, insecure data transportation, trust inconsistencies among stakeholders, incompliant data storing mechanisms, etc. Edge-servers often suffer because of their lightweight computation capacity to stamp out unauthorized data or instructions, which in essence, makes them exposed to attackers. When attackers target edge servers while transporting data using traditional PKI-rendered trusts, consortium blockchain (CBC) offers proven techniques to transfer and maintain those sensitive data securely. With the recent improvement of edge machine learning, edge devices can filter malicious data at their end which largely motivates us to institute a Blockchain and AI aligned APT detection system. The unique contributions of the paper include efficient APT detection at the edge and transparent recording of the detection history in an immutable blockchain ledger. In line with that, the certificateless data transfer mechanism boost trust among collaborators and ensure an economical and sustainable mechanism after eliminating existing certificate authority. Finally, the edge-compliant storage technique facilitates efficient predictive maintenance. The respective experimental outcomes reveal that the proposed technique outperforms the other competing systems and models.
       


## February
### [Semantic of Cloud Computing services for Time Series workflows](https://arxiv.org/abs/2202.00609)

**Authors:**
Manuel Parra-Royón, Francisco Baldan, Ghislain Atemezing, J. M. Benitez

**Abstract:**
Time series (TS) are present in many fields of knowledge, research, and engineering. The processing and analysis of TS are essential in order to extract knowledge from the data and to tackle forecasting or predictive maintenance tasks among others The modeling of TS is a challenging task, requiring high statistical expertise as well as outstanding knowledge about the application of Data Mining(DM) and Machine Learning (ML) methods. The overall work with TS is not limited to the linear application of several techniques, but is composed of an open workflow of methods and tests. These workflow, developed mainly on programming languages, are complicated to execute and run effectively on different systems, including Cloud Computing (CC) environments. The adoption of CC can facilitate the integration and portability of services allowing to adopt solutions towards services Internet Technologies (IT) industrialization. The definition and description of workflow services for TS open up a new set of possibilities regarding the reduction of complexity in the deployment of this type of issues in CC environments. In this sense, we have designed an effective proposal based on semantic modeling (or vocabulary) that provides the full description of workflow for Time Series modeling as a CC service. Our proposal includes a broad spectrum of the most extended operations, accommodating any workflow applied to classification, regression, or clustering problems for Time Series, as well as including evaluation measures, information, tests, or machine learning algorithms among others.
       


### [CycleGAN for Undamaged-to-Damaged Domain Translation for Structural Health Monitoring and Damage Detection](https://arxiv.org/abs/2202.07831)

**Authors:**
Furkan Luleci, F. Necati Catbas, Onur Avci

**Abstract:**
The recent advances in the data science field in the last few decades have benefitted many other fields including Structural Health Monitoring (SHM). Particularly, Artificial Intelligence (AI) such as Machine Learning (ML) and Deep Learning (DL) methods for vibration-based damage diagnostics of civil structures has been utilized extensively due to the observed high performances in learning from data. Along with diagnostics, damage prognostics is also vitally important for estimating the remaining useful life of civil structures. Currently, AI-based data-driven methods used for damage diagnostics and prognostics centered on historical data of the structures and require a substantial amount of data for prediction models. Although some of these methods are generative-based models, they are used to perform ML or DL tasks such as classification, regression, clustering, etc. after learning the distribution of the data. In this study, a variant of Generative Adversarial Networks (GAN), Cycle-Consistent Wasserstein Deep Convolutional GAN with Gradient Penalty (CycleWDCGAN-GP) model is developed to investigate the "transition of structural dynamic signature from an undamaged-to-damaged state" and "if this transition can be employed for predictive damage detection". The outcomes of this study demonstrate that the proposed model can accurately generate damaged responses from undamaged responses or vice versa. In other words, it will be possible to understand the damaged condition while the structure is still in a healthy (undamaged) condition or vice versa with the proposed methodology. This will enable a more proactive approach in overseeing the life-cycle performance as well as in predicting the remaining useful life of structures.
       


### [Remaining Useful Life Prediction Using Temporal Deep Degradation Network for Complex Machinery with Attention-based Feature Extraction](https://arxiv.org/abs/2202.10916)

**Authors:**
Yuwen Qin, Ningbo Cai, Chen Gao, Yadong Zhang, Yonghong Cheng, Xin Chen

**Abstract:**
The precise estimate of remaining useful life (RUL) is vital for the prognostic analysis and predictive maintenance that can significantly reduce failure rate and maintenance costs. The degradation-related features extracted from the sensor streaming data with neural networks can dramatically improve the accuracy of the RUL prediction. The Temporal deep degradation network (TDDN) model is proposed to make the RUL prediction with the degradation-related features given by the one-dimensional convolutional neural network (1D CNN) feature extraction and attention mechanism. 1D CNN is used to extract the temporal features from the streaming sensor data. Temporal features have monotonic degradation trends from the fluctuating raw sensor streaming data. Attention mechanism can improve the RUL prediction performance by capturing the fault characteristics and the degradation development with the attention weights. The performance of the TDDN model is evaluated on the public C-MAPSS dataset and compared with the existing methods. The results show that the TDDN model can achieve the best RUL prediction accuracy in complex conditions compared to current machine learning models. The degradation-related features extracted from the high-dimension sensor streaming data demonstrate the clear degradation trajectories and degradation stages that enable TDDN to predict the turbofan-engine RUL accurately and efficiently.
       


## March
### [Predicting Bearings' Degradation Stages for Predictive Maintenance in the Pharmaceutical Industry](https://arxiv.org/abs/2203.03259)

**Authors:**
Dovile Juodelyte, Veronika Cheplygina, Therese Graversen, Philippe Bonnet

**Abstract:**
In the pharmaceutical industry, the maintenance of production machines must be audited by the regulator. In this context, the problem of predictive maintenance is not when to maintain a machine, but what parts to maintain at a given point in time. The focus shifts from the entire machine to its component parts and prediction becomes a classification problem. In this paper, we focus on rolling-elements bearings and we propose a framework for predicting their degradation stages automatically. Our main contribution is a k-means bearing lifetime segmentation method based on high-frequency bearing vibration signal embedded in a latent low-dimensional subspace using an AutoEncoder. Given high-frequency vibration data, our framework generates a labeled dataset that is used to train a supervised model for bearing degradation stage detection. Our experimental results, based on the FEMTO Bearing dataset, show that our framework is scalable and that it provides reliable and actionable predictions for a range of different bearings.
       


### [Battery Cloud with Advanced Algorithms](https://arxiv.org/abs/2203.03737)

**Authors:**
Xiaojun Li, David Jauernig, Mengzhu Gao, Trevor Jones

**Abstract:**
A Battery Cloud or cloud battery management system leverages the cloud computational power and data storage to improve battery safety, performance, and economy. This work will present the Battery Cloud that collects measured battery data from electric vehicles and energy storage systems. Advanced algorithms are applied to improve battery performance. Using remote vehicle data, we train and validate an artificial neural network to estimate pack SOC during vehicle charging. The strategy is then tested on vehicles. Furthermore, high accuracy and onboard battery state of health estimation methods for electric vehicles are developed based on the differential voltage (DVA) and incremental capacity analysis (ICA). Using cycling data from battery cells at various temperatures, we extract the charging cycles and calculate the DVA and ICA curves, from which multiple features are extracted, analyzed, and eventually used to estimate the state of health. For battery safety, a data-driven thermal anomaly detection method is developed. The method can detect unforeseen anomalies such as thermal runaways at the very early stage. With the further development of the internet of things, more and more battery data will be available. Potential applications of battery cloud also include areas such as battery manufacture, recycling, and electric vehicle battery swap.
       


### [Quantitative characterisation of the layered structure within lithium-ion batteries using ultrasonic resonance](https://arxiv.org/abs/2203.04149)

**Authors:**
Ming Huang, Niall Kirkaldy, Yan Zhao, Yatish Patel, Frederic Cegla, Bo Lan

**Abstract:**
Lithium-ion batteries (LIBs) are becoming an important energy storage solution to achieve carbon neutrality, but it remains challenging to characterise their internal states for the assurance of performance, durability and safety. This work reports a simple but powerful non-destructive characterisation technique, based on the formation of ultrasonic resonance from the repetitive layers within LIBs. A physical model is developed from the ground up, to interpret the results from standard experimental ultrasonic measurement setups. As output, the method delivers a range of critical pieces of information about the inner structure of LIBs, such as the number of layers, the average thicknesses of electrodes, the image of internal layers, and the states of charge variations across individual layers. This enables the quantitative tracking of internal cell properties, potentially providing new means of quality control during production processes, and tracking the states of health and charge during operation.
       


### [Extending life of Lithium-ion battery systems by embracing heterogeneities via an optimal control-based active balancing strategy](https://arxiv.org/abs/2203.04226)

**Authors:**
Vahid Azimi, Anirudh Allam, Simona Onori

**Abstract:**
This paper formulates and solves a multi-objective fast charging-minimum degradation optimal control problem (OCP) for a lithium-ion battery module made of series-connected cells equipped with an active balancing circuitry. The cells in the module are subject to heterogeneity induced by manufacturing defects and non-uniform operating conditions. Each cell is expressed via a coupled nonlinear electrochemical, thermal, and aging model and the direct collocation approach is employed to transcribe the OCP into a nonlinear programming problem (NLP). The proposed OCP is formulated under two different schemes of charging operation: (i) same-charging-time (OCP-SCT) and (ii) different-charging-time (OCP-DCT). The former assumes simultaneous charging of all cells irrespective of their initial conditions, whereas the latter allows for different charging times of the cells to account for heterogeneous initial conditions. The problem is solved for a module with two series-connected cells with intrinsic heterogeneity among them in terms of state of charge and state of health. Results show that the OCP-DCT scheme provides more flexibility to deal with heterogeneity, boasting of lower temperature increase, charging current amplitudes, and degradation. Finally, comparison with the common practice of constant current (CC) charging over a long-term cycling operation shows that promising savings, in terms of retained capacity, are attainable under both the control (OCP-SCT and OCP-DCT) schemes.
       


### [Identifying the root cause of cable network problems with machine learning](https://arxiv.org/abs/2203.06989)

**Authors:**
Georg Heiler, Thassilo Gadermaier, Thomas Haider, Allan Hanbury, Peter Filzmoser

**Abstract:**
Good quality network connectivity is ever more important. For hybrid fiber coaxial (HFC) networks, searching for upstream high noise in the past was cumbersome and time-consuming. Even with machine learning due to the heterogeneity of the network and its topological structure, the task remains challenging. We present the automation of a simple business rule (largest change of a specific value) and compare its performance with state-of-the-art machine-learning methods and conclude that the precision@1 can be improved by 2.3 times. As it is best when a fault does not occur in the first place, we secondly evaluate multiple approaches to forecast network faults, which would allow performing predictive maintenance on the network.
       


### [Evaluation of websites of state public health agencies during the COVID-19 pandemic demonstrating the degree of effort to design for accessibility](https://arxiv.org/abs/2203.07201)

**Authors:**
Arunkumar Pennathur, Amirmasoud Momenipour, Priyadarshini Pennathur, Brandon Murphy

**Abstract:**
Since the beginning of the pandemic, every state public health agency in the United States has created and maintained a website dedicated to COVID 19. Our goal was to evaluate these websites for conformity to accessibility guidelines. Specifically, we assessed, on a scale of increasing levels of accessibility compliance requirements, the results of the efforts made by website developers to incorporate and meet accessibility compliance criteria. We focused on homepages and vaccine pages in 49 states. For this study, we used the automated AChecker tool to assess conformance to the WCAG 2.0 guidelines at A, AA and AAA levels of conformance, and conformance with the Section 508c standard. We also manually rated, on a scale 0 (none) to 3 (highest), the specific accessibility features, if any, that web developers had included on the pages. We found that accessibility violations were prevalent across states but to varying degrees for a specific accessibility criterion. Although violations were detected in all 4 POUR accessibility principles, the most number of known violations occurred in meeting the perceivability and operability principles. Most violations in 508c guidelines occurred in not providing functional text in scripting languages and in not providing text equivalents for nontext. The degree of effort and conformance significantly varied between states; a majority of states exhibited a lower degree of effort, while a few attempted innovative ways to enhance accessibility on their websites. The efforts seemed to focus on meeting the minimum threshold. It is not clear if websites were designed proactively for accessibility.
       


### [Machine Learning based Data Driven Diagnostic and Prognostic Approach for Laser Reliability Enhancement](https://arxiv.org/abs/2203.11728)

**Authors:**
Khouloud Abdelli, Helmut Griesser, Stephan Pachnicke

**Abstract:**
In this paper, a data-driven diagnostic and prognostic approach based on machine learning is proposed to detect laser failure modes and to predict the remaining useful life (RUL) of a laser during its operation. We present an architecture of the proposed cognitive predictive maintenance framework and demonstrate its effectiveness using synthetic data.
       


### [A Hybrid CNN-LSTM Approach for Laser Remaining Useful Life Prediction](https://arxiv.org/abs/2203.12415)

**Authors:**
Khouloud Abdelli, Helmut Griesser, Stephan Pachnicke

**Abstract:**
A hybrid prognostic model based on convolutional neural networks (CNN) and long short-term memory (LSTM) is proposed to predict the laser remaining useful life (RUL). The experimental results show that it outperforms the conventional methods.
       


### [DeepALM: Holistic Optical Network Monitoring based on Machine Learning](https://arxiv.org/abs/2203.13596)

**Authors:**
Joo Yeon Cho, Jose-Juan Pedreno-Manresa, Sai Kireet Patri, Khouloud Abdelli, Carsten Tropschug, Jim Zou, Piotr Rydlichowski

**Abstract:**
We demonstrate a machine learning-based optical network monitoring system which can integrate fiber monitoring, predictive maintenance of optical hardware, and security information management in a single solution.
       


### [Slow-varying Dynamics Assisted Temporal Capsule Network for Machinery Remaining Useful Life Estimation](https://arxiv.org/abs/2203.16373)

**Authors:**
Yan Qin, Chau Yuen, Yimin Shao, Bo Qin, Xiaoli Li

**Abstract:**
Capsule network (CapsNet) acts as a promising alternative to the typical convolutional neural network, which is the dominant network to develop the remaining useful life (RUL) estimation models for mechanical equipment. Although CapsNet comes with an impressive ability to represent the entities' hierarchical relationships through a high-dimensional vector embedding, it fails to capture the long-term temporal correlation of run-to-failure time series measured from degraded mechanical equipment. On the other hand, the slow-varying dynamics, which reveals the low-frequency information hidden in mechanical dynamical behaviour, is overlooked in the existing RUL estimation models, limiting the utmost ability of advanced networks. To address the aforementioned concerns, we propose a Slow-varying Dynamics assisted Temporal CapsNet (SD-TemCapsNet) to simultaneously learn the slow-varying dynamics and temporal dynamics from measurements for accurate RUL estimation. First, in light of the sensitivity of fault evolution, slow-varying features are decomposed from normal raw data to convey the low-frequency components corresponding to the system dynamics. Next, the long short-term memory (LSTM) mechanism is introduced into CapsNet to capture the temporal correlation of time series. To this end, experiments conducted on an aircraft engine and a milling machine verify that the proposed SD-TemCapsNet outperforms the mainstream methods. In comparison with CapsNet, the estimation accuracy of the aircraft engine with four different scenarios has been improved by 10.17%, 24.97%, 3.25%, and 13.03% concerning the index root mean squared error, respectively. Similarly, the estimation accuracy of the milling machine has been improved by 23.57% compared to LSTM and 19.54% compared to CapsNet.
       


## April
### [When to Classify Events in Open Times Series?](https://arxiv.org/abs/2204.00392)

**Authors:**
Youssef Achenchabe, Alexis Bondu, Antoine Cornuéjols, Vincent Lemaire

**Abstract:**
In numerous applications, for instance in predictive maintenance, there is a pression to predict events ahead of time with as much accuracy as possible while not delaying the decision unduly. This translates in the optimization of a trade-off between earliness and accuracy of the decisions, that has been the subject of research for time series of finite length and with a unique label. And this has led to powerful algorithms for Early Classification of Time Series (ECTS). This paper, for the first time, investigates such a trade-off when events of different classes occur in a streaming fashion, with no predefined end. In the Early Classification in Open Time Series problem (ECOTS), the task is to predict events, i.e. their class and time interval, at the moment that optimizes the accuracy vs. earliness trade-off. Interestingly, we find that ECTS algorithms can be sensibly adapted in a principled way to this new problem. We illustrate our methodology by transforming two state-of-the-art ECTS algorithms for the ECOTS scenario. Among the wide variety of applications that this new approach opens up, we develop a predictive maintenance use case that optimizes alarm triggering times, thus demonstrating the power of this new approach.
       


### [T4PdM: a Deep Neural Network based on the Transformer Architecture for Fault Diagnosis of Rotating Machinery](https://arxiv.org/abs/2204.03725)

**Authors:**
Erick Giovani Sperandio Nascimento, Julian Santana Liang, Ilan Sousa Figueiredo, Lilian Lefol Nani Guarieiro

**Abstract:**
Deep learning and big data algorithms have become widely used in industrial applications to optimize several tasks in many complex systems. Particularly, deep learning model for diagnosing and prognosing machinery health has leveraged predictive maintenance (PdM) to be more accurate and reliable in decision making, in this way avoiding unnecessary interventions, machinery accidents, and environment catastrophes. Recently, Transformer Neural Networks have gained notoriety and have been increasingly the favorite choice for Natural Language Processing (NLP) tasks. Thus, given their recent major achievements in NLP, this paper proposes the development of an automatic fault classifier model for predictive maintenance based on a modified version of the Transformer architecture, namely T4PdM, to identify multiple types of faults in rotating machinery. Experimental results are developed and presented for the MaFaulDa and CWRU databases. T4PdM was able to achieve an overall accuracy of 99.98% and 98% for both datasets, respectively. In addition, the performance of the proposed model is compared to other previously published works. It has demonstrated the superiority of the model in detecting and classifying faults in rotating industrial machinery. Therefore, the proposed Transformer-based model can improve the performance of machinery fault analysis and diagnostic processes and leverage companies to a new era of the Industry 4.0. In addition, this methodology can be adapted to any other task of time series classification.
       


### [Prognostic classification based on random convolutional kernel](https://arxiv.org/abs/2204.04527)

**Authors:**
Zekun Wu, Kaiwei Wu

**Abstract:**
Assessing the health status (HS) of system/component has long been a challenging task in the prognostic and health management (PHM) study. Differed from other regression based prognostic task such as predicting the remaining useful life, the HS assessment is essentially a multi class classificatIon problem. To address this issue, we introduced the random convolutional kernel-based approach, the RandOm Convolutional KErnel Transforms (ROCKET) and its latest variant MiniROCKET, in the paper. We implement ROCKET and MiniROCKET on the NASA's CMPASS dataset and assess the turbine fan engine's HS with the multi-sensor time-series data. Both methods show great accuracy when tackling the HS assessment task. More importantly, they demonstrate considerably efficiency especially compare with the deep learning-based method. We further reveal that the feature generated by random convolutional kernel can be combined with other classifiers such as support vector machine (SVM) and linear discriminant analysis (LDA). The newly constructed method maintains the high efficiency and outperforms all the other deop neutal network models in classification accuracy.
       


### [SAL-CNN: Estimate the Remaining Useful Life of Bearings Using Time-frequency Information](https://arxiv.org/abs/2204.05045)

**Authors:**
Bingguo Liu, Zhuo Gao, Binghui Lu, Hangcheng Dong, Zeru An

**Abstract:**
In modern industrial production, the prediction ability of the remaining useful life (RUL) of bearings directly affects the safety and stability of the system. Traditional methods require rigorous physical modeling and perform poorly for complex systems. In this paper, an end-to-end RUL prediction method is proposed, which uses short-time Fourier transform (STFT) as preprocessing. Considering the time correlation of signal sequences, a long and short-term memory network is designed in CNN, incorporating the convolutional block attention module, and understanding the decision-making process of the network from the interpretability level. Experiments were carried out on the 2012PHM dataset and compared with other methods, and the results proved the effectiveness of the method.
       


### [A Variational Autoencoder for Heterogeneous Temporal and Longitudinal Data](https://arxiv.org/abs/2204.09369)

**Authors:**
Mine Öğretir, Siddharth Ramchandran, Dimitrios Papatheodorou, Harri Lähdesmäki

**Abstract:**
The variational autoencoder (VAE) is a popular deep latent variable model used to analyse high-dimensional datasets by learning a low-dimensional latent representation of the data. It simultaneously learns a generative model and an inference network to perform approximate posterior inference. Recently proposed extensions to VAEs that can handle temporal and longitudinal data have applications in healthcare, behavioural modelling, and predictive maintenance. However, these extensions do not account for heterogeneous data (i.e., data comprising of continuous and discrete attributes), which is common in many real-life applications. In this work, we propose the heterogeneous longitudinal VAE (HL-VAE) that extends the existing temporal and longitudinal VAEs to heterogeneous data. HL-VAE provides efficient inference for high-dimensional datasets and includes likelihood models for continuous, count, categorical, and ordinal data while accounting for missing observations. We demonstrate our model's efficacy through simulated as well as clinical datasets, and show that our proposed model achieves competitive performance in missing value imputation and predictive accuracy.
       


### [A two-level machine learning framework for predictive maintenance: comparison of learning formulations](https://arxiv.org/abs/2204.10083)

**Authors:**
Valentin Hamaide, Denis Joassin, Lauriane Castin, François Glineur

**Abstract:**
Predicting incoming failures and scheduling maintenance based on sensors information in industrial machines is increasingly important to avoid downtime and machine failure. Different machine learning formulations can be used to solve the predictive maintenance problem. However, many of the approaches studied in the literature are not directly applicable to real-life scenarios. Indeed, many of those approaches usually either rely on labelled machine malfunctions in the case of classification and fault detection, or rely on finding a monotonic health indicator on which a prediction can be made in the case of regression and remaining useful life estimation, which is not always feasible. Moreover, the decision-making part of the problem is not always studied in conjunction with the prediction phase. This paper aims to design and compare different formulations for predictive maintenance in a two-level framework and design metrics that quantify both the failure detection performance as well as the timing of the maintenance decision. The first level is responsible for building a health indicator by aggregating features using a learning algorithm. The second level consists of a decision-making system that can trigger an alarm based on this health indicator. Three degrees of refinements are compared in the first level of the framework, from simple threshold-based univariate predictive technique to supervised learning methods based on the remaining time before failure. We choose to use the Support Vector Machine (SVM) and its variations as the common algorithm used in all the formulations. We apply and compare the different strategies on a real-world rotating machine case study and observe that while a simple model can already perform well, more sophisticated refinements enhance the predictions for well-chosen parameters.
       


### [Multi-Component Optimization and Efficient Deployment of Neural-Networks on Resource-Constrained IoT Hardware](https://arxiv.org/abs/2204.10183)

**Authors:**
Bharath Sudharsan, Dineshkumar Sundaram, Pankesh Patel, John G. Breslin, Muhammad Intizar Ali, Schahram Dustdar, Albert Zomaya, Rajiv Ranjan

**Abstract:**
The majority of IoT devices like smartwatches, smart plugs, HVAC controllers, etc., are powered by hardware with a constrained specification (low memory, clock speed and processor) which is insufficient to accommodate and execute large, high-quality models. On such resource-constrained devices, manufacturers still manage to provide attractive functionalities (to boost sales) by following the traditional approach of programming IoT devices/products to collect and transmit data (image, audio, sensor readings, etc.) to their cloud-based ML analytics platforms. For decades, this online approach has been facing issues such as compromised data streams, non-real-time analytics due to latency, bandwidth constraints, costly subscriptions, recent privacy issues raised by users and the GDPR guidelines, etc. In this paper, to enable ultra-fast and accurate AI-based offline analytics on resource-constrained IoT devices, we present an end-to-end multi-component model optimization sequence and open-source its implementation. Researchers and developers can use our optimization sequence to optimize high memory, computation demanding models in multiple aspects in order to produce small size, low latency, low-power consuming models that can comfortably fit and execute on resource-constrained hardware. The experimental results show that our optimization components can produce models that are; (i) 12.06 x times compressed; (ii) 0.13% to 0.27% more accurate; (iii) Orders of magnitude faster unit inference at 0.06 ms. Our optimization sequence is generic and can be applied to any state-of-the-art models trained for anomaly detection, predictive maintenance, robotics, voice recognition, and machine vision.
       


### [Logistic-ELM: A Novel Fault Diagnosis Method for Rolling Bearings](https://arxiv.org/abs/2204.11845)

**Authors:**
Zhenhua Tan, Jingyu Ning, Kai Peng, Zhenche Xia, Danke Wu

**Abstract:**
The fault diagnosis of rolling bearings is a critical technique to realize predictive maintenance for mechanical condition monitoring. In real industrial systems, the main challenges for the fault diagnosis of rolling bearings pertain to the accuracy and real-time requirements. Most existing methods focus on ensuring the accuracy, and the real-time requirement is often neglected. In this paper, considering both requirements, we propose a novel fast fault diagnosis method for rolling bearings, based on extreme learning machine (ELM) and logistic mapping, named logistic-ELM. First, we identify 14 kinds of time-domain features from the original vibration signals according to mechanical vibration principles and adopt the sequential forward selection (SFS) strategy to select optimal features from them to ensure the basic predictive accuracy and efficiency. Next, we propose the logistic-ELM for fast fault classification, where the biases in ELM are omitted and the random input weights are replaced by the chaotic logistic mapping sequence which involves a higher uncorrelation to obtain more accurate results with fewer hidden neurons. We conduct extensive experiments on the rolling bearing vibration signal dataset of the Case Western Reserve University (CWRU) Bearing Data Centre. The experimental results show that the proposed approach outperforms existing SOTA comparison methods in terms of the predictive accuracy, and the highest accuracy is 100% in seven separate sub data environments. The relevant code is publicly available at https://github.com/TAN-OpenLab/logistic-ELM.
       


### [An Explainable Regression Framework for Predicting Remaining Useful Life of Machines](https://arxiv.org/abs/2204.13574)

**Authors:**
Talhat Khan, Kashif Ahmad, Jebran Khan, Imran Khan, Nasir Ahmad

**Abstract:**
Prediction of a machine's Remaining Useful Life (RUL) is one of the key tasks in predictive maintenance. The task is treated as a regression problem where Machine Learning (ML) algorithms are used to predict the RUL of machine components. These ML algorithms are generally used as a black box with a total focus on the performance without identifying the potential causes behind the algorithms' decisions and their working mechanism. We believe, the performance (in terms of Mean Squared Error (MSE), etc.,) alone is not enough to build the trust of the stakeholders in ML prediction rather more insights on the causes behind the predictions are needed. To this aim, in this paper, we explore the potential of Explainable AI (XAI) techniques by proposing an explainable regression framework for the prediction of machines' RUL. We also evaluate several ML algorithms including classical and Neural Networks (NNs) based solutions for the task. For the explanations, we rely on two model agnostic XAI methods namely Local Interpretable Model-Agnostic Explanations (LIME) and Shapley Additive Explanations (SHAP). We believe, this work will provide a baseline for future research in the domain.
       


## May
### [On off-line and on-line Bayesian filtering for uncertainty quantification of structural deterioration](https://arxiv.org/abs/2205.03478)

**Authors:**
Antonios Kamariotis, Luca Sardi, Iason Papaioannou, Eleni Chatzi, Daniel Straub

**Abstract:**
Data-informed predictive maintenance planning largely relies on stochastic deterioration models. Monitoring information can be utilized to update sequentially the knowledge on time-invariant deterioration model parameters either within an off-line (batch) or an on-line (recursive) Bayesian framework. With a focus on the quantification of the full parameter uncertainty, we review, adapt and investigate selected Bayesian filters for parameter estimation: an on-line particle filter, an on-line iterated batch importance sampling filter, which performs Markov chain Monte Carlo (MCMC) move steps, and an off-line MCMC-based sequential Monte Carlo filter. A Gaussian mixture model is used to approximate the posterior distribution within the resampling process in all three filters. Two numerical examples serve as the basis for a comparative assessment of off-line and on-line Bayesian estimation of time-invariant deterioration model parameters. The first case study considers a low-dimensional, nonlinear, non-Gaussian probabilistic fatigue crack growth model that is updated with sequential crack monitoring measurements. The second high-dimensional, linear, Gaussian case study employs a random field to model corrosion deterioration across a beam, which is updated with sequential measurements from sensors. The numerical investigations provide insights into the performance of off-line and on-line filters in terms of the accuracy of posterior estimates and the computational cost, when applied to problems of different nature, increasing dimensionality and varying sensor information amount. Importantly, they show that a tailored implementation of the on-line particle filter proves competitive with the computationally demanding MCMC-based filters. Suggestions on the choice of the appropriate method in function of problem characteristics are provided.
       


### [State of Health Estimation of Lithium-Ion Batteries in Vehicle-to-Grid Applications Using Recurrent Neural Networks for Learning the Impact of Degradation Stress Factors](https://arxiv.org/abs/2205.07561)

**Authors:**
Kotub Uddin, James Schofield, W. Dhammika Widanage

**Abstract:**
This work presents an effective state of health indicator to indicate lithium-ion battery degradation based on a long short-term memory (LSTM) recurrent neural network (RNN) coupled with a sliding-window. The developed LSTM RNN is able to capture the underlying long-term dependencies of degraded cell capacity on battery degradation stress factors. The learning performance was robust when there was sufficient training data, with an error of < 5% if more than 1.15 years worth of data was supplied for training.
       


### [Predictive Maintenance using Machine Learning](https://arxiv.org/abs/2205.09402)

**Authors:**
Archit P. Kane, Ashutosh S. Kore, Advait N. Khandale, Sarish S. Nigade, Pranjali P. Joshi

**Abstract:**
Predictive maintenance (PdM) is a concept, which is implemented to effectively manage maintenance plans of the assets by predicting their failures with data driven techniques. In these scenarios, data is collected over a certain period of time to monitor the state of equipment. The objective is to find some correlations and patterns that can help predict and ultimately prevent failures. Equipment in manufacturing industry are often utilized without a planned maintenance approach. Such practise frequently results in unexpected downtime, owing to certain unexpected failures. In scheduled maintenance, the condition of the manufacturing equipment is checked after fixed time interval and if any fault occurs, the component is replaced to avoid unexpected equipment stoppages. On the flip side, this leads to increase in time for which machine is non-functioning and cost of carrying out the maintenance. The emergence of Industry 4.0 and smart systems have led to increasing emphasis on predictive maintenance (PdM) strategies that can reduce the cost of downtime and increase the availability (utilization rate) of manufacturing equipment. PdM also has the potential to bring about new sustainable practices in manufacturing by fully utilizing the useful lives of components.
       


### [Trends in Workplace Wearable Technologies and Connected-Worker Solutions for Next-Generation Occupational Safety, Health, and Productivity](https://arxiv.org/abs/2205.11740)

**Authors:**
Vishal Patel, Austin Chesmore, Christopher M. Legner, Santosh Pandey

**Abstract:**
The workplace influences the safety, health, and productivity of workers at multiple levels. To protect and promote total worker health, smart hardware, and software tools have emerged for the identification, elimination, substitution, and control of occupational hazards. Wearable devices enable constant monitoring of individual workers and the environment, whereas connected worker solutions provide contextual information and decision support. Here, the recent trends in commercial workplace technologies to monitor and manage occupational risks, injuries, accidents, and diseases are reviewed. Workplace safety wearables for safe lifting, ergonomics, hazard identification, sleep monitoring, fatigue management, and heat and cold stress are discussed. Examples of workplace productivity wearables for asset tracking, augmented reality, gesture and motion control, brain wave sensing, and work stress management are given. Workplace health wearables designed for work-related musculoskeletal disorders, functional movement disorders, respiratory hazards, cardiovascular health, outdoor sun exposure, and continuous glucose monitoring are shown. Connected worker platforms are discussed with information about the architecture, system modules, intelligent operations, and industry applications. Predictive analytics provide contextual information about occupational safety risks, resource allocation, equipment failure, and predictive maintenance. Altogether, these examples highlight the ground-level benefits of real-time visibility about frontline workers, work environment, distributed assets, workforce efficiency, and safety compliance
       


### [Maximum Mean Discrepancy on Exponential Windows for Online Change Detection](https://arxiv.org/abs/2205.12706)

**Authors:**
Florian Kalinke, Marco Heyden, Georg Gntuni, Edouard Fouché, Klemens Böhm

**Abstract:**
Detecting changes is of fundamental importance when analyzing data streams and has many applications, e.g., in predictive maintenance, fraud detection, or medicine. A principled approach to detect changes is to compare the distributions of observations within the stream to each other via hypothesis testing. Maximum mean discrepancy (MMD), a (semi-)metric on the space of probability distributions, provides powerful non-parametric two-sample tests on kernel-enriched domains. In particular, MMD is able to detect any disparity between distributions under mild conditions. However, classical MMD estimators suffer from a quadratic runtime complexity, which renders their direct use for change detection in data streams impractical. In this article, we propose a new change detection algorithm, called Maximum Mean Discrepancy on Exponential Windows (MMDEW), that combines the benefits of MMD with an efficient computation based on exponential windows. We prove that MMDEW enjoys polylogarithmic runtime and logarithmic memory complexity and show empirically that it outperforms the state of the art on benchmark data streams.
       


### [Survival Analysis on Structured Data using Deep Reinforcement Learning](https://arxiv.org/abs/2205.14331)

**Authors:**
Renith G, Harikrishna Warrier, Yogesh Gupta

**Abstract:**
Survival analysis is playing a major role in manufacturing sector by analyzing occurrence of any unwanted event based on the input data. Predictive maintenance, which is a part of survival analysis, helps to find any device failure based on the current incoming data from different sensor or any equipment. Deep learning techniques were used to automate the predictive maintenance problem to some extent, but they are not very helpful in predicting the device failure for the input data which the algorithm had not learned. Since neural network predicts the output based on previous learned input features, it cannot perform well when there is more variation in input features. Performance of the model is degraded with the occurrence of changes in input data and finally the algorithm fails in predicting the device failure. This problem can be solved by our proposed method where the algorithm can predict the device failure more precisely than the existing deep learning algorithms. The proposed solution involves implementation of Deep Reinforcement Learning algorithm called Double Deep Q Network (DDQN) for classifying the device failure based on the input features. The algorithm is capable of learning different variation of the input feature and is robust in predicting whether the device will fail or not based on the input data. The proposed DDQN model is trained with limited or lesser amount of input data. The trained model predicted larger amount of test data efficiently and performed well compared to other deep learning and machine learning models.
       


## June
### [Anomaly Detection and Inter-Sensor Transfer Learning on Smart Manufacturing Datasets](https://arxiv.org/abs/2206.06355)

**Authors:**
Mustafa Abdallah, Byung-Gun Joung, Wo Jae Lee, Charilaos Mousoulis, John W. Sutherland, Saurabh Bagchi

**Abstract:**
Smart manufacturing systems are being deployed at a growing rate because of their ability to interpret a wide variety of sensed information and act on the knowledge gleaned from system observations. In many cases, the principal goal of the smart manufacturing system is to rapidly detect (or anticipate) failures to reduce operational cost and eliminate downtime. This often boils down to detecting anomalies within the sensor date acquired from the system. The smart manufacturing application domain poses certain salient technical challenges. In particular, there are often multiple types of sensors with varying capabilities and costs. The sensor data characteristics change with the operating point of the environment or machines, such as, the RPM of the motor. The anomaly detection process therefore has to be calibrated near an operating point. In this paper, we analyze four datasets from sensors deployed from manufacturing testbeds. We evaluate the performance of several traditional and ML-based forecasting models for predicting the time series of sensor data. Then, considering the sparse data from one kind of sensor, we perform transfer learning from a high data rate sensor to perform defect type classification. Taken together, we show that predictive failure classification can be achieved, thus paving the way for predictive maintenance.
       


### [A Machine Learning-based Digital Twin for Electric Vehicle Battery Modeling](https://arxiv.org/abs/2206.08080)

**Authors:**
Khaled Sidahmed Sidahmed Alamin, Yukai Chen, Enrico Macii, Massimo Poncino, Sara Vinco

**Abstract:**
The widespread adoption of Electric Vehicles (EVs) is limited by their reliance on batteries with presently low energy and power densities compared to liquid fuels and are subject to aging and performance deterioration over time. For this reason, monitoring the battery State Of Charge (SOC) and State Of Health (SOH) during the EV lifetime is a very relevant problem. This work proposes a battery digital twin structure designed to accurately reflect battery dynamics at the run time. To ensure a high degree of correctness concerning non-linear phenomena, the digital twin relies on data-driven models trained on traces of battery evolution over time: a SOH model, repeatedly executed to estimate the degradation of maximum battery capacity, and a SOC model, retrained periodically to reflect the impact of aging. The proposed digital twin structure will be exemplified on a public dataset to motivate its adoption and prove its effectiveness, with high accuracy and inference and retraining times compatible with onboard execution.
       


### [Propagation of chaos for a stochastic particle system modelling epidemics](https://arxiv.org/abs/2206.09862)

**Authors:**
Alessandro Ciallella, Mario Pulvirenti, Sergio Simonella

**Abstract:**
We consider a simple stochastic $N$-particle system, already studied by the same authors in \cite{CPS21}, representing different populations of agents. Each agent has a label describing his state of health. We show rigorously that, in the limit $N \to \infty$, propagation of chaos holds, leading to a set of kinetic equations which are a spatially inhomogeneous version of the classical SIR model. We improve a similar result obtained in \cite{CPS21} by using here a different coupling technique, which makes the analysis simpler, more natural and transparent.
       


### [The Digital Twin Landscape at the Crossroads of Predictive Maintenance, Machine Learning and Physics Based Modeling](https://arxiv.org/abs/2206.10462)

**Authors:**
Brian Kunzer, Mario Berges, Artur Dubrawski

**Abstract:**
The concept of a digital twin has exploded in popularity over the past decade, yet confusion around its plurality of definitions, its novelty as a new technology, and its practical applicability still exists, all despite numerous reviews, surveys, and press releases. The history of the term digital twin is explored, as well as its initial context in the fields of product life cycle management, asset maintenance, and equipment fleet management, operations, and planning. A definition for a minimally viable framework to utilize a digital twin is also provided based on seven essential elements. A brief tour through DT applications and industries where DT methods are employed is also outlined. The application of a digital twin framework is highlighted in the field of predictive maintenance, and its extensions utilizing machine learning and physics based modeling. Employing the combination of machine learning and physics based modeling to form hybrid digital twin frameworks, may synergistically alleviate the shortcomings of each method when used in isolation. Key challenges of implementing digital twin models in practice are additionally discussed. As digital twin technology experiences rapid growth and as it matures, its great promise to substantially enhance tools and solutions for intelligent upkeep of complex equipment, are expected to materialize.
       


### [Human-in-the-Loop Large-Scale Predictive Maintenance of Workstations](https://arxiv.org/abs/2206.11574)

**Authors:**
Alexander Nikitin, Samuel Kaski

**Abstract:**
Predictive maintenance (PdM) is the task of scheduling maintenance operations based on a statistical analysis of the system's condition. We propose a human-in-the-loop PdM approach in which a machine learning system predicts future problems in sets of workstations (computers, laptops, and servers). Our system interacts with domain experts to improve predictions and elicit their knowledge. In our approach, domain experts are included in the loop not only as providers of correct labels, as in traditional active learning, but as a source of explicit decision rule feedback. The system is automated and designed to be easily extended to novel domains, such as maintaining workstations of several organizations. In addition, we develop a simulator for reproducible experiments in a controlled environment and deploy the system in a large-scale case of real-life workstations PdM with thousands of workstations for dozens of companies.
       


### [Interpretable Hidden Markov Model-Based Deep Reinforcement Learning Hierarchical Framework for Predictive Maintenance of Turbofan Engines](https://arxiv.org/abs/2206.13433)

**Authors:**
Ammar N. Abbas, Georgios Chasparis, John D. Kelleher

**Abstract:**
An open research question in deep reinforcement learning is how to focus the policy learning of key decisions within a sparse domain. This paper emphasizes combining the advantages of inputoutput hidden Markov models and reinforcement learning towards interpretable maintenance decisions. We propose a novel hierarchical-modeling methodology that, at a high level, detects and interprets the root cause of a failure as well as the health degradation of the turbofan engine, while, at a low level, it provides the optimal replacement policy. It outperforms the baseline performance of deep reinforcement learning methods applied directly to the raw data or when using a hidden Markov model without such a specialized hierarchy. It also provides comparable performance to prior work, however, with the additional benefit of interpretability.
       


### [Multi-Fault Diagnosis Of Industrial Rotating Machines Using Data-Driven Approach: A Review Of Two Decades Of Research](https://arxiv.org/abs/2206.14153)

**Authors:**
Shreyas Gawde, Shruti Patil, Satish Kumar, Pooja Kamat, Ketan Kotecha, Ajith Abraham

**Abstract:**
Industry 4.0 is an era of smart manufacturing. Manufacturing is impossible without the use of machinery. Majority of these machines comprise rotating components and are called rotating machines. The engineers' top priority is to maintain these critical machines to reduce the unplanned shutdown and increase the useful life of machinery. Predictive maintenance (PDM) is the current trend of smart maintenance. The challenging task in PDM is to diagnose the type of fault. With Artificial Intelligence (AI) advancement, data-driven approach for predictive maintenance is taking a new flight towards smart manufacturing. Several researchers have published work related to fault diagnosis in rotating machines, mainly exploring a single type of fault. However, a consolidated review of literature that focuses more on multi-fault diagnosis of rotating machines is lacking. There is a need to systematically cover all the aspects right from sensor selection, data acquisition, feature extraction, multi-sensor data fusion to the systematic review of AI techniques employed in multi-fault diagnosis. In this regard, this paper attempts to achieve the same by implementing a systematic literature review on a Data-driven approach for multi-fault diagnosis of Industrial Rotating Machines using Preferred Reporting Items for Systematic Reviews and Meta-Analysis (PRISMA) method. The PRISMA method is a collection of guidelines for the composition and structure of systematic reviews and other meta-analyses. This paper identifies the foundational work done in the field and gives a comparative study of different aspects related to multi-fault diagnosis of industrial rotating machines. The paper also identifies the major challenges, research gap. It gives solutions using recent advancements in AI in implementing multi-fault diagnosis, giving a strong base for future research in this field.
       


## July
### [Predicting Li-ion Battery Cycle Life with LSTM RNN](https://arxiv.org/abs/2207.03687)

**Authors:**
Pengcheng Xu, Yunfeng Lu

**Abstract:**
Efficient and accurate remaining useful life prediction is a key factor for reliable and safe usage of lithium-ion batteries. This work trains a long short-term memory recurrent neural network model to learn from sequential data of discharge capacities at various cycles and voltages and to work as a cycle life predictor for battery cells cycled under different conditions. Using experimental data of first 60 - 80 cycles, our model achieves promising prediction accuracy on test sets of around 80 samples.
       


### [Data-Driven Thermal Modelling for Anomaly Detection in Electric Vehicle Charging Stations](https://arxiv.org/abs/2207.05431)

**Authors:**
Pere Izquierdo Gómez, Alberto Barragan Moreno, Jun Lin, Tomislav Dragičević

**Abstract:**
The rapid growth of the electric vehicle (EV) sector is giving rise to many infrastructural challenges. One such challenge is its requirement for the widespread development of EV charging stations which must be able to provide large amounts of power in an on-demand basis. This can cause large stresses on the electrical and electronic components of the charging infrastructure - negatively affecting its reliability as well as leading to increased maintenance and operation costs. This paper proposes a human-interpretable data-driven method for anomaly detection in EV charging stations, aiming to provide information for the condition monitoring and predictive maintenance of power converters within such a station. To this end, a model of a high-efficiency EV charging station is used to simulate the thermal behaviour of EV charger power converter modules, creating a data set for the training of neural network models. These machine learning models are then employed for the identification of anomalous performance.
       


### [A Benchmark dataset for predictive maintenance](https://arxiv.org/abs/2207.05466)

**Authors:**
Bruno Veloso, João Gama, Rita P. Ribeiro, Pedro M. Pereira

**Abstract:**
The paper describes the MetroPT data set, an outcome of a eXplainable Predictive Maintenance (XPM) project with an urban metro public transportation service in Porto, Portugal. The data was collected in 2022 that aimed to evaluate machine learning methods for online anomaly detection and failure prediction. By capturing several analogic sensor signals (pressure, temperature, current consumption), digital signals (control signals, discrete signals), and GPS information (latitude, longitude, and speed), we provide a dataset that can be easily used to evaluate online machine learning methods. This dataset contains some interesting characteristics and can be a good benchmark for predictive maintenance models.
       


### [Digital Twin-based Intrusion Detection for Industrial Control Systems](https://arxiv.org/abs/2207.09999)

**Authors:**
Seba Anna Varghese, Alireza Dehlaghi Ghadim, Ali Balador, Zahra Alimadadi, Panos Papadimitratos

**Abstract:**
Digital twins have recently gained significant interest in simulation, optimization, and predictive maintenance of Industrial Control Systems (ICS). Recent studies discuss the possibility of using digital twins for intrusion detection in industrial systems. Accordingly, this study contributes to a digital twin-based security framework for industrial control systems, extending its capabilities for simulation of attacks and defense mechanisms. Four types of process-aware attack scenarios are implemented on a standalone open-source digital twin of an industrial filling plant: command injection, network Denial of Service (DoS), calculated measurement modification, and naive measurement modification. A stacked ensemble classifier is proposed as the real-time intrusion detection, based on the offline evaluation of eight supervised machine learning algorithms. The designed stacked model outperforms previous methods in terms of F1-Score and accuracy, by combining the predictions of various algorithms, while it can detect and classify intrusions in near real-time (0.1 seconds). This study also discusses the practicality and benefits of the proposed digital twin-based security framework.
       


### [Security and Safety Aspects of AI in Industry Applications](https://arxiv.org/abs/2207.10809)

**Author:**
Hans Dermot Doran

**Abstract:**
In this relatively informal discussion-paper we summarise issues in the domains of safety and security in machine learning that will affect industry sectors in the next five to ten years. Various products using neural network classification, most often in vision related applications but also in predictive maintenance, have been researched and applied in real-world applications in recent years. Nevertheless, reports of underlying problems in both safety and security related domains, for instance adversarial attacks have unsettled early adopters and are threatening to hinder wider scale adoption of this technology. The problem for real-world applicability lies in being able to assess the risk of applying these technologies. In this discussion-paper we describe the process of arriving at a machine-learnt neural network classifier pointing out safety and security vulnerabilities in that workflow, citing relevant research where appropriate.
       


## August
### [Vacuum Circuit Breaker Closing Time Key Moments Detection via Vibration Monitoring: A Run-to-Failure Study](https://arxiv.org/abs/2208.07607)

**Authors:**
Chi-Ching Hsu, Gaetan Frusque, Mahir Muratovic, Christian M. Franck, Olga Fink

**Abstract:**
Circuit breakers (CBs) play an important role in modern society because they make the power transmission and distribution systems reliable and resilient. Therefore, it is important to maintain their reliability and to monitor their operation. A key to ensure a reliable operation of CBs is to monitor their condition. In this work, we performed an accelerated life testing for mechanical failures of a vacuum circuit breaker (VCB) by performing close-open operations continuously until failure. We recorded data for each operation and made the collected run-to-failure dataset publicly available. In our experiments, the VCB operated more than 26000 close-open operations without current load with the time span of five months. The run-to-failure long-term monitoring enables us to monitor the evolution of the VCB condition and the degradation over time. To monitor CB condition, closing time is one of the indicators, which is usually measured when the CB is taken out of operation and is completely disconnected from the network. We propose an algorithm that enables to infer the same information on the closing time from a non-intrusive sensor. By utilizing the short-time energy (STE) of the vibration signal, it is possible to identify the key moments when specific events happen including the time when the latch starts to move, and the closing time. The effectiveness of the proposed algorithm is evaluated on the VCB dataset and is also compared to the binary segmentation (BS) change point detection algorithm. This research highlights the potential for continuous online condition monitoring, which is the basis for applying future predictive maintenance strategies.
       


### [LAMA-Net: Unsupervised Domain Adaptation via Latent Alignment and Manifold Learning for RUL Prediction](https://arxiv.org/abs/2208.08388)

**Authors:**
Manu Joseph, Varchita Lalwani

**Abstract:**
Prognostics and Health Management (PHM) is an emerging field which has received much attention from the manufacturing industry because of the benefits and efficiencies it brings to the table. And Remaining Useful Life (RUL) prediction is at the heart of any PHM system. Most recent data-driven research demand substantial volumes of labelled training data before a performant model can be trained under the supervised learning paradigm. This is where Transfer Learning (TL) and Domain Adaptation (DA) methods step in and make it possible for us to generalize a supervised model to other domains with different data distributions with no labelled data. In this paper, we propose \textit{LAMA-Net}, an encoder-decoder based model (Transformer) with an induced bottleneck, Latent Alignment using Maximum Mean Discrepancy (MMD) and manifold learning is proposed to tackle the problem of Unsupervised Homogeneous Domain Adaptation for RUL prediction. \textit{LAMA-Net} is validated using the C-MAPSS Turbofan Engine dataset by NASA and compared against other state-of-the-art techniques for DA. The results suggest that the proposed method offers a promising approach to perform domain adaptation in RUL prediction. Code will be made available once the paper comes out of review.
       


### [On an Application of Generative Adversarial Networks on Remaining Lifetime Estimation](https://arxiv.org/abs/2208.08666)

**Authors:**
G. Tsialiamanis, D. Wagg, N. Dervilis, K. Worden

**Abstract:**
A major problem of structural health monitoring (SHM) has been the prognosis of damage and the definition of the remaining useful life of a structure. Both tasks depend on many parameters, many of which are often uncertain. Many models have been developed for the aforementioned tasks but they have been either deterministic or stochastic with the ability to take into account only a restricted amount of past states of the structure. In the current work, a generative model is proposed in order to make predictions about the damage evolution of structures. The model is able to perform in a population-based SHM (PBSHM) framework, to take into account many past states of the damaged structure, to incorporate uncertainties in the modelling process and to generate potential damage evolution outcomes according to data acquired from a structure. The algorithm is tested on a simulated damage evolution example and the results reveal that it is able to provide quite confident predictions about the remaining useful life of structures within a population.
       


### [Transfer Learning-based State of Health Estimation for Lithium-ion Battery with Cycle Synchronization](https://arxiv.org/abs/2208.11204)

**Authors:**
Kate Qi Zhou, Yan Qin, Chau Yuen

**Abstract:**
Accurately estimating a battery's state of health (SOH) helps prevent battery-powered applications from failing unexpectedly. With the superiority of reducing the data requirement of model training for new batteries, transfer learning (TL) emerges as a promising machine learning approach that applies knowledge learned from a source battery, which has a large amount of data. However, the determination of whether the source battery model is reasonable and which part of information can be transferred for SOH estimation are rarely discussed, despite these being critical components of a successful TL. To address these challenges, this paper proposes an interpretable TL-based SOH estimation method by exploiting the temporal dynamic to assist transfer learning, which consists of three parts. First, with the help of dynamic time warping, the temporal data from the discharge time series are synchronized, yielding the warping path of the cycle-synchronized time series responsible for capacity degradation over cycles. Second, the canonical variates retrieved from the spatial path of the cycle-synchronized time series are used for distribution similarity analysis between the source and target batteries. Third, when the distribution similarity is within the predefined threshold, a comprehensive target SOH estimation model is constructed by transferring the common temporal dynamics from the source SOH estimation model and compensating the errors with a residual model from the target battery. Through a widely-used open-source benchmark dataset, the estimation error of the proposed method evaluated by the root mean squared error is as low as 0.0034 resulting in a 77% accuracy improvement compared with existing methods.
       


### [System Resilience through Health Monitoring and Reconfiguration](https://arxiv.org/abs/2208.14525)

**Authors:**
Ion Matei, Wiktor Piotrowski, Alexandre Perez, Johan de Kleer, Jorge Tierno, Wendy Mungovan, Vance Turnewitsch

**Abstract:**
We demonstrate an end-to-end framework to improve the resilience of man-made systems to unforeseen events. The framework is based on a physics-based digital twin model and three modules tasked with real-time fault diagnosis, prognostics and reconfiguration. The fault diagnosis module uses model-based diagnosis algorithms to detect and isolate faults and generates interventions in the system to disambiguate uncertain diagnosis solutions. We scale up the fault diagnosis algorithm to the required real-time performance through the use of parallelization and surrogate models of the physics-based digital twin. The prognostics module tracks the fault progressions and trains the online degradation models to compute remaining useful life of system components. In addition, we use the degradation models to assess the impact of the fault progression on the operational requirements. The reconfiguration module uses PDDL-based planning endowed with semantic attachments to adjust the system controls so that the fault impact on the system operation is minimized. We define a resilience metric and use the example of a fuel system model to demonstrate how the metric improves with our framework.
       


## September
### [A Transferable Multi-stage Model with Cycling Discrepancy Learning for Lithium-ion Battery State of Health Estimation](https://arxiv.org/abs/2209.00190)

**Authors:**
Yan Qin, Chau Yuen, Xunyuan Yin, Biao Huang

**Abstract:**
As a significant ingredient regarding health status, data-driven state-of-health (SOH) estimation has become dominant for lithium-ion batteries (LiBs). To handle data discrepancy across batteries, current SOH estimation models engage in transfer learning (TL), which reserves apriori knowledge gained through reusing partial structures of the offline trained model. However, multiple degradation patterns of a complete life cycle of a battery make it challenging to pursue TL. The concept of the stage is introduced to describe the collection of continuous cycles that present a similar degradation pattern. A transferable multi-stage SOH estimation model is proposed to perform TL across batteries in the same stage, consisting of four steps. First, with identified stage information, raw cycling data from the source battery are reconstructed into the phase space with high dimensions, exploring hidden dynamics with limited sensors. Next, domain invariant representation across cycles in each stage is proposed through cycling discrepancy subspace with reconstructed data. Third, considering the unbalanced discharge cycles among different stages, a switching estimation strategy composed of a lightweight model with the long short-term memory network and a powerful model with the proposed temporal capsule network is proposed to boost estimation accuracy. Lastly, an updating scheme compensates for estimation errors when the cycling consistency of target batteries drifts. The proposed method outperforms its competitive algorithms in various transfer tasks for a run-to-failure benchmark with three batteries.
       


### [An Optimized and Safety-aware Maintenance Framework: A Case Study on Aircraft Engine](https://arxiv.org/abs/2209.02678)

**Authors:**
Muhammad Ziyad, Kenrick Tjandra, Zulvah, Mushonnifun Faiz Sugihartanto, Mansur Arief

**Abstract:**
The COVID-19 pandemic has recently exacerbated the fierce competition in the transportation businesses. The airline industry took one of the biggest hits as the closure of international borders forced aircraft operators to suspend their international routes, keeping aircraft on the ground without generating revenues while at the same time still requiring adequate maintenance. To maintain their operational sustainability, finding a good balance between cost reductions measure and safety standards fulfillment, including its maintenance procedure, becomes critical. This paper proposes an AI-assisted predictive maintenance scheme that synthesizes prognostics modeling and simulation-based optimization to help airlines decide their optimal engine maintenance approach. The proposed method enables airlines to utilize their diagnostics measurements and operational settings to design a more customized maintenance strategy that takes engine operations conditions into account. Our numerical experiments on the proposed approach resulted in significant cost savings without compromising the safety standards. The experiments also show that maintenance strategies tailored to the failure mode and operational settings (that our framework enables) yield 13% more cost savings than generic optimal maintenance strategies. The generality of our proposed framework allows the extension to other intelligent, safety-critical transportation systems.
       


### [Transfer Learning and Vision Transformer based State-of-Health prediction of Lithium-Ion Batteries](https://arxiv.org/abs/2209.05253)

**Authors:**
Pengyu Fu, Liang Chu, Zhuoran Hou, Jincheng Hu, Yanjun Huang, Yuanjian Zhang

**Abstract:**
In recent years, significant progress has been made in transportation electrification. And lithium-ion batteries (LIB), as the main energy storage devices, have received widespread attention. Accurately predicting the state of health (SOH) can not only ease the anxiety of users about the battery life but also provide important information for the management of the battery. This paper presents a prediction method for SOH based on Vision Transformer (ViT) model. First, discrete charging data of a predefined voltage range is used as an input data matrix. Then, the cycle features of the battery are captured by the ViT which can obtain the global features, and the SOH is obtained by combining the cycle features with the full connection (FC) layer. At the same time, transfer learning (TL) is introduced, and the prediction model based on source task battery training is further fine-tuned according to the early cycle data of the target task battery to provide an accurate prediction. Experiments show that our method can obtain better feature expression compared with existing deep learning methods so that better prediction effect and transfer effect can be achieved.
       


### [A Hybrid Deep Learning Model-based Remaining Useful Life Estimation for Reed Relay with Degradation Pattern Clustering](https://arxiv.org/abs/2209.06429)

**Authors:**
Chinthaka Gamanayake, Yan Qin, Chau Yuen, Lahiru Jayasinghe, Dominique-Ea Tan, Jenny Low

**Abstract:**
Reed relay serves as the fundamental component of functional testing, which closely relates to the successful quality inspection of electronics. To provide accurate remaining useful life (RUL) estimation for reed relay, a hybrid deep learning network with degradation pattern clustering is proposed based on the following three considerations. First, multiple degradation behaviors are observed for reed relay, and hence a dynamic time wrapping-based $K$-means clustering is offered to distinguish degradation patterns from each other. Second, although proper selections of features are of great significance, few studies are available to guide the selection. The proposed method recommends operational rules for easy implementation purposes. Third, a neural network for remaining useful life estimation (RULNet) is proposed to address the weakness of the convolutional neural network (CNN) in capturing temporal information of sequential data, which incorporates temporal correlation ability after high-level feature representation of convolutional operation. In this way, three variants of RULNet are constructed with health indicators, features with self-organizing map, or features with curve fitting. Ultimately, the proposed hybrid model is compared with the typical baseline models, including CNN and long short-term memory network (LSTM), through a practical reed relay dataset with two distinct degradation manners. The results from both degradation cases demonstrate that the proposed method outperforms CNN and LSTM regarding the index root mean squared error.
       


### [A Temporal Anomaly Detection System for Vehicles utilizing Functional Working Groups and Sensor Channels](https://arxiv.org/abs/2209.06828)

**Authors:**
Subash Neupane, Ivan A. Fernandez, Wilson Patterson, Sudip Mittal, Shahram Rahimi

**Abstract:**
A modern vehicle fitted with sensors, actuators, and Electronic Control Units (ECUs) can be divided into several operational subsystems called Functional Working Groups (FWGs). Examples of these FWGs include the engine system, transmission, fuel system, brakes, etc. Each FWG has associated sensor-channels that gauge vehicular operating conditions. This data rich environment is conducive to the development of Predictive Maintenance (PdM) technologies. Undercutting various PdM technologies is the need for robust anomaly detection models that can identify events or observations which deviate significantly from the majority of the data and do not conform to a well defined notion of normal vehicular operational behavior. In this paper, we introduce the Vehicle Performance, Reliability, and Operations (VePRO) dataset and use it to create a multi-phased approach to anomaly detection. Utilizing Temporal Convolution Networks (TCN), our anomaly detection system can achieve 96% detection accuracy and accurately predicts 91% of true anomalies. The performance of our anomaly detection system improves when sensor channels from multiple FWGs are utilized.
       


### [Leveraging the Potential of Novel Data in Power Line Communication of Electricity Grids](https://arxiv.org/abs/2209.12693)

**Authors:**
Christoph Balada, Max Bondorf, Sheraz Ahmed, Andreas Dengela, Markus Zdrallek

**Abstract:**
Electricity grids have become an essential part of daily life, even if they are often not noticed in everyday life. We usually only become particularly aware of this dependence by the time the electricity grid is no longer available. However, significant changes, such as the transition to renewable energy (photovoltaic, wind turbines, etc.) and an increasing number of energy consumers with complex load profiles (electric vehicles, home battery systems, etc.), pose new challenges for the electricity grid. To address these challenges, we propose two first-of-its-kind datasets based on measurements in a broadband powerline communications (PLC) infrastructure. Both datasets FiN-1 and FiN-2, were collected during real practical use in a part of the German low-voltage grid that supplies around 4.4 million people and show more than 13 billion datapoints collected by more than 5100 sensors. In addition, we present different use cases in asset management, grid state visualization, forecasting, predictive maintenance, and novelty detection to highlight the benefits of these types of data. For these applications, we particularly highlight the use of novel machine learning architectures to extract rich information from real-world data that cannot be captured using traditional approaches. By publishing the first large-scale real-world dataset, we aim to shed light on the previously largely unrecognized potential of PLC data and emphasize machine-learning-based research in low-voltage distribution networks by presenting a variety of different use cases.
       


## October
### [Tracking changes using Kullback-Leibler divergence for the continual learning](https://arxiv.org/abs/2210.04865)

**Authors:**
Sebastián Basterrech, Michal Woźniak

**Abstract:**
Recently, continual learning has received a lot of attention. One of the significant problems is the occurrence of \emph{concept drift}, which consists of changing probabilistic characteristics of the incoming data. In the case of the classification task, this phenomenon destabilizes the model's performance and negatively affects the achieved prediction quality. Most current methods apply statistical learning and similarity analysis over the raw data. However, similarity analysis in streaming data remains a complex problem due to time limitation, non-precise values, fast decision speed, scalability, etc. This article introduces a novel method for monitoring changes in the probabilistic distribution of multi-dimensional data streams. As a measure of the rapidity of changes, we analyze the popular Kullback-Leibler divergence. During the experimental study, we show how to use this metric to predict the concept drift occurrence and understand its nature. The obtained results encourage further work on the proposed methods and its application in the real tasks where the prediction of the future appearance of concept drift plays a crucial role, such as predictive maintenance.
       


### [A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID](https://arxiv.org/abs/2210.07317)

**Authors:**
Hong Yang, Travis Desell

**Abstract:**
This paper presents the largest publicly available, non-simulated, fleet-wide aircraft flight recording and maintenance log data for use in predicting part failure and maintenance need. We present 31,177 hours of flight data across 28,935 flights, which occur relative to 2,111 unplanned maintenance events clustered into 36 types of maintenance issues. Flights are annotated as before or after maintenance, with some flights occurring on the day of maintenance. Collecting data to evaluate predictive maintenance systems is challenging because it is difficult, dangerous, and unethical to generate data from compromised aircraft. To overcome this, we use the National General Aviation Flight Information Database (NGAFID), which contains flights recorded during regular operation of aircraft, and maintenance logs to construct a part failure dataset. We use a novel framing of Remaining Useful Life (RUL) prediction and consider the probability that the RUL of a part is greater than 2 days. Unlike previous datasets generated with simulations or in laboratory settings, the NGAFID Aviation Maintenance Dataset contains real flight records and maintenance logs from different seasons, weather conditions, pilots, and flight patterns. Additionally, we provide Python code to easily download the dataset and a Colab environment to reproduce our benchmarks on three different models. Our dataset presents a difficult challenge for machine learning researchers and a valuable opportunity to test and develop prognostic health management methods
       


### [A four compartment epidemic model with retarded transition rates](https://arxiv.org/abs/2210.09912)

**Authors:**
Teo Granger, Thomas M. Michelitsch, Michael Bestehorn, Alejandro P. Riascos, Bernard A. Collet

**Abstract:**
We study an epidemic model for a constant population by taking into account four compartments of the individuals characterizing their states of health. Each individual is in one of the compartments susceptible (S); incubated - infected yet not infectious (C), infected and infectious (I), and recovered - immune (R). An infection is 'visible' only when an individual is in state I. Upon infection, an individual performs the transition pathway S to C to I to R to S remaining in each compartments C, I, and R for certain random waiting times, respectively. The waiting times for each compartment are independent and drawn from specific probability density functions (PDFs) introducing memory into the model. We derive memory evolution equations involving convolutions (time derivatives of general fractional type). We obtain formulae for the endemic equilibrium and a condition of its existence for cases when the waiting time PDFs have existing means. We analyze the stability of healthy and endemic equilibria and derive conditions for which the endemic state becomes oscillatory (Hopf) unstable. We implement a simple multiple random walker's approach (microscopic model of Brownian motion of Z independent walkers) with random SCIRS waiting times into computer simulations. Infections occur with a certain probability by collisions of walkers in compartments I and S. We compare the endemic states predicted in the macroscopic model with the numerical results of the simulations and find accordance of high accuracy. We conclude that a simple random walker's approach offers an appropriate microscopic description for the macroscopic model.
       


### [DIICAN: Dual Time-scale State-Coupled Co-estimation of SOC, SOH and RUL for Lithium-Ion Batteries](https://arxiv.org/abs/2210.11941)

**Authors:**
Ningbo Cai, Yuwen Qin, Xin Chen, Kai Wu

**Abstract:**
Accurate co-estimations of battery states, such as state-of-charge (SOC), state-of-health (SOH,) and remaining useful life (RUL), are crucial to the battery management systems to assure safe and reliable management. Although the external properties of the battery charge with the aging degree, batteries' degradation mechanism shares similar evolving patterns. Since batteries are complicated chemical systems, these states are highly coupled with intricate electrochemical processes. A state-coupled co-estimation method named Deep Inter and Intra-Cycle Attention Network (DIICAN) is proposed in this paper to estimate SOC, SOH, and RUL, which organizes battery measurement data into the intra-cycle and inter-cycle time scales. And to extract degradation-related features automatically and adapt to practical working conditions, the convolutional neural network is applied. The state degradation attention unit is utilized to extract the battery state evolution pattern and evaluate the battery degradation degree. To account for the influence of battery aging on the SOC estimation, the battery degradation-related state is incorporated in the SOC estimation for capacity calibration. The DIICAN method is validated on the Oxford battery dataset. The experimental results show that the proposed method can achieve SOH and RUL co-estimation with high accuracy and effectively improve SOC estimation accuracy for the whole lifespan.
       


## November
### [An IoT Cloud and Big Data Architecture for the Maintenance of Home Appliances](https://arxiv.org/abs/2211.02627)

**Authors:**
Pedro Chaves, Tiago Fonseca, Luis Lino Ferreira, Bernardo Cabral, Orlando Sousa, Andre Oliveira, Jorge Landeck

**Abstract:**
Billions of interconnected Internet of Things (IoT) sensors and devices collect tremendous amounts of data from real-world scenarios. Big data is generating increasing interest in a wide range of industries. Once data is analyzed through compute-intensive Machine Learning (ML) methods, it can derive critical business value for organizations. Powerfulplatforms are essential to handle and process such massive collections of information cost-effectively and conveniently. This work introduces a distributed and scalable platform architecture that can be deployed for efficient real-world big data collection and analytics. The proposed system was tested with a case study for Predictive Maintenance of Home Appliances, where current and vibration sensors with high acquisition frequency were connected to washing machines and refrigerators. The introduced platform was used to collect, store, and analyze the data. The experimental results demonstrated that the presented system could be advantageous for tackling real-world IoT scenarios in a cost-effective and local approach.
       


### [A Machine Learning-based Framework for Predictive Maintenance of Semiconductor Laser for Optical Communication](https://arxiv.org/abs/2211.02842)

**Authors:**
Khouloud Abdelli, Helmut Griesser, Stephan Pachnicke

**Abstract:**
Semiconductor lasers, one of the key components for optical communication systems, have been rapidly evolving to meet the requirements of next generation optical networks with respect to high speed, low power consumption, small form factor etc. However, these demands have brought severe challenges to the semiconductor laser reliability. Therefore, a great deal of attention has been devoted to improving it and thereby ensuring reliable transmission. In this paper, a predictive maintenance framework using machine learning techniques is proposed for real-time heath monitoring and prognosis of semiconductor laser and thus enhancing its reliability. The proposed approach is composed of three stages: i) real-time performance degradation prediction, ii) degradation detection, and iii) remaining useful life (RUL) prediction. First of all, an attention based gated recurrent unit (GRU) model is adopted for real-time prediction of performance degradation. Then, a convolutional autoencoder is used to detect the degradation or abnormal behavior of a laser, given the predicted degradation performance values. Once an abnormal state is detected, a RUL prediction model based on attention-based deep learning is utilized. Afterwards, the estimated RUL is input for decision making and maintenance planning. The proposed framework is validated using experimental data derived from accelerated aging tests conducted for semiconductor tunable lasers. The proposed approach achieves a very good degradation performance prediction capability with a small root mean square error (RMSE) of 0.01, a good anomaly detection accuracy of 94.24% and a better RUL estimation capability compared to the existing ML-based laser RUL prediction models.
       


### [A Comprehensive Survey of Regression Based Loss Functions for Time Series Forecasting](https://arxiv.org/abs/2211.02989)

**Authors:**
Aryan Jadon, Avinash Patil, Shruti Jadon

**Abstract:**
Time Series Forecasting has been an active area of research due to its many applications ranging from network usage prediction, resource allocation, anomaly detection, and predictive maintenance. Numerous publications published in the last five years have proposed diverse sets of objective loss functions to address cases such as biased data, long-term forecasting, multicollinear features, etc. In this paper, we have summarized 14 well-known regression loss functions commonly used for time series forecasting and listed out the circumstances where their application can aid in faster and better model convergence. We have also demonstrated how certain categories of loss functions perform well across all data sets and can be considered as a baseline objective function in circumstances where the distribution of the data is unknown. Our code is available at GitHub: https://github.com/aryan-jadon/Regression-Loss-Functions-in-Time-Series-Forecasting-Tensorflow.
       


### [Federated Learning for Autoencoder-based Condition Monitoring in the Industrial Internet of Things](https://arxiv.org/abs/2211.07619)

**Authors:**
Soeren Becker, Kevin Styp-Rekowski, Oliver Vincent Leon Stoll, Odej Kao

**Abstract:**
Enabled by the increasing availability of sensor data monitored from production machinery, condition monitoring and predictive maintenance methods are key pillars for an efficient and robust manufacturing production cycle in the Industrial Internet of Things. The employment of machine learning models to detect and predict deteriorating behavior by analyzing a variety of data collected across several industrial environments shows promising results in recent works, yet also often requires transferring the sensor data to centralized servers located in the cloud. Moreover, although collaborating and sharing knowledge between industry sites yields large benefits, especially in the area of condition monitoring, it is often prohibited due to data privacy issues. To tackle this situation, we propose an Autoencoder-based Federated Learning method utilizing vibration sensor data from rotating machines, that allows for a distributed training on edge devices, located on-premise and close to the monitored machines. Preserving data privacy and at the same time exonerating possibly unreliable network connections of remote sites, our approach enables knowledge transfer across organizational boundaries, without sharing the monitored data. We conducted an evaluation utilizing two real-world datasets as well as multiple testbeds and the results indicate that our method enables a competitive performance compared to previous results, while significantly reducing the resource and network utilization.
       


### [Data fusion techniques for fault diagnosis of industrial machines: a survey](https://arxiv.org/abs/2211.09551)

**Authors:**
Amir Eshaghi Chaleshtori, Abdollah aghaie

**Abstract:**
In the Engineering discipline, predictive maintenance techniques play an essential role in improving system safety and reliability of industrial machines. Due to the adoption of crucial and emerging detection techniques and big data analytics tools, data fusion approaches are gaining popularity. This article thoroughly reviews the recent progress of data fusion techniques in predictive maintenance, focusing on their applications in machinery fault diagnosis. In this review, the primary objective is to classify existing literature and to report the latest research and directions to help researchers and professionals to acquire a clear understanding of the thematic area. This paper first summarizes fundamental data-fusion strategies for fault diagnosis. Then, a comprehensive investigation of the different levels of data fusion was conducted on fault diagnosis of industrial machines. In conclusion, a discussion of data fusion-based fault diagnosis challenges, opportunities, and future trends are presented.
       


### [Circuit Design for Predictive Maintenance](https://arxiv.org/abs/2211.10248)

**Authors:**
Taner Dosluoglu, Martin MacDonald

**Abstract:**
Industry 4.0 has become a driver for the entire manufacturing industry. Smart systems have enabled 30% productivity increases and predictive maintenance has been demonstrated to provide a 50% reduction in machine downtime. So far, the solution has been based on data analytics which has resulted in a proliferation of sensing technologies and infrastructure for data acquisition, transmission and processing. At the core of factory operation and automation are circuits that control and power factory equipment, innovative circuit design has the potential to address many system integration challenges. We present a new circuit design approach based on circuit level artificial intelligence solutions, integrated within control and calibration functional blocks during circuit design, improving the predictability and adaptability of each component for predictive maintenance. This approach is envisioned to encourage the development of new EDA tools such as automatic digital shadow generation and product lifecycle models, that will help identification of circuit parameters that adequately define the operating conditions for dynamic prediction and fault detection. Integration of a supplementary artificial intelligence block within the control loop is considered for capturing non-linearities and gain/bandwidth constraints of the main controller and identifying changes in the operating conditions beyond the response of the controller. System integration topics are discussed regarding integration within OPC Unified Architecture and predictive maintenance interfaces, providing real-time updates to the digital shadow that help maintain an accurate, virtual replica model of the physical system.
       


### [PreMa: Predictive Maintenance of Solenoid Valve in Real-Time at Embedded Edge-Level](https://arxiv.org/abs/2211.12326)

**Authors:**
Prajwal BN, Harsha Yelchuri, Vishwanath Shastry, T. V. Prabhakar

**Abstract:**
In industrial process automation, sensors (pressure, temperature, etc.), controllers, and actuators (solenoid valves, electro-mechanical relays, circuit breakers, motors, etc.) make sure that production lines are working under the pre-defined conditions. When these systems malfunction or sometimes completely fail, alerts have to be generated in real-time to make sure not only production quality is not compromised but also safety of humans and equipment is assured. In this work, we describe the construction of a smart and real-time edge-based electronic product called PreMa, which is basically a sensor for monitoring the health of a Solenoid Valve (SV). PreMa is compact, low power, easy to install, and cost effective. It has data fidelity and measurement accuracy comparable to signals captured using high end equipment. The smart solenoid sensor runs TinyML, a compact version of TensorFlow (a.k.a. TFLite) machine learning framework. While fault detection inferencing is in-situ, model training uses mobile phones to accomplish the `on-device' training. Our product evaluation shows that the sensor is able to differentiate between the distinct types of faults. These faults include: (a) Spool stuck (b) Spring failure and (c) Under voltage. Furthermore, the product provides maintenance personnel, the remaining useful life (RUL) of the SV. The RUL provides assistance to decide valve replacement or otherwise. We perform an extensive evaluation on optimizing metrics related to performance of the entire system (i.e. embedded platform and the neural network model). The proposed implementation is such that, given any electro-mechanical actuator with similar transient response to that of the SV, the system is capable of condition monitoring, hence presenting a first of its kind generic infrastructure.
       


### [EDGAR: Embedded Detection of Gunshots by AI in Real-time](https://arxiv.org/abs/2211.14073)

**Author:**
Nathan Morsa

**Abstract:**
Electronic shot counters allow armourers to perform preventive and predictive maintenance based on quantitative measurements, improving reliability, reducing the frequency of accidents, and reducing maintenance costs. To answer a market pressure for both low lead time to market and increased customisation, we aim to solve the shot detection and shot counting problem in a generic way through machine learning.
  In this study, we describe a method allowing one to construct a dataset with minimal labelling effort by only requiring the total number of shots fired in a time series. To our knowledge, this is the first study to propose a technique, based on learning from label proportions, that is able to exploit these weak labels to derive an instance-level classifier able to solve the counting problem and the more general discrimination problem. We also show that this technique can be deployed in heavily constrained microcontrollers while still providing hard real-time (<100ms) inference. We evaluate our technique against a state-of-the-art unsupervised algorithm and show a sizeable improvement, suggesting that the information from the weak labels is successfully leveraged. Finally, we evaluate our technique against human-generated state-of-the-art algorithms and show that it provides comparable performance and significantly outperforms them in some offline and real-world benchmarks.
       


## December
### [Data-Driven Prognosis of Failure Detection and Prediction of Lithium-ion Batteries](https://arxiv.org/abs/2212.01236)

**Authors:**
Hamed Sadegh Kouhestani, Lin Liu, Ruimin Wang, Abhijit Chandra

**Abstract:**
Battery prognostics and health management predictive models are essential components of safety and reliability protocols in battery management system frameworks. Overall, developing a robust and efficient battery model that aligns with the current literature is a useful step in ensuring the safety of battery function. For this purpose, a multi-physics, multi-scale deterministic data-driven prognosis (DDP) is proposed that only relies on in situ measurements of data and estimates the failure based on the curvature information extracted from the system. Unlike traditional applications that require explicit expression of conservation principle, the proposed method devices a local conservation functional in the neighborhood of each data point which is represented as the minimization of curvature in the system. By eliminating the need for offline training, the method can predict the onset of instability for a variety of systems over a prediction horizon. The prediction horizon to prognosticate the instability, alternatively, is considered as the remaining useful life (RUL) metric. The framework is then employed to analyze the health status of Li-ion batteries. Based on the results, it has demonstrated that the DDP technique can accurately predict the onset of failure of Li-ion batteries.
       


### [Enhanced Gaussian Process Dynamical Models with Knowledge Transfer for Long-term Battery Degradation Forecasting](https://arxiv.org/abs/2212.01609)

**Authors:**
Wei W. Xing, Ziyang Zhang, Akeel A. Shah

**Abstract:**
Predicting the end-of-life or remaining useful life of batteries in electric vehicles is a critical and challenging problem, predominantly approached in recent years using machine learning to predict the evolution of the state-of-health during repeated cycling. To improve the accuracy of predictive estimates, especially early in the battery lifetime, a number of algorithms have incorporated features that are available from data collected by battery management systems. Unless multiple battery data sets are used for a direct prediction of the end-of-life, which is useful for ball-park estimates, such an approach is infeasible since the features are not known for future cycles. In this paper, we develop a highly-accurate method that can overcome this limitation, by using a modified Gaussian process dynamical model (GPDM). We introduce a kernelised version of GPDM for a more expressive covariance structure between both the observable and latent coordinates. We combine the approach with transfer learning to track the future state-of-health up to end-of-life. The method can incorporate features as different physical observables, without requiring their values beyond the time up to which data is available. Transfer learning is used to improve learning of the hyperparameters using data from similar batteries. The accuracy and superiority of the approach over modern benchmarks algorithms including a Gaussian process model and deep convolutional and recurrent networks are demonstrated on three data sets, particularly at the early stages of the battery lifetime.
       


### [Towards a Taxonomy for the Use of Synthetic Data in Advanced Analytics](https://arxiv.org/abs/2212.02622)

**Authors:**
Peter Kowalczyk, Giacomo Welsch, Frédéric Thiesse

**Abstract:**
The proliferation of deep learning techniques led to a wide range of advanced analytics applications in important business areas such as predictive maintenance or product recommendation. However, as the effectiveness of advanced analytics naturally depends on the availability of sufficient data, an organization's ability to exploit the benefits might be restricted by limited data or likewise data access. These challenges could force organizations to spend substantial amounts of money on data, accept constrained analytics capacities, or even turn into a showstopper for analytics projects. Against this backdrop, recent advances in deep learning to generate synthetic data may help to overcome these barriers. Despite its great potential, however, synthetic data are rarely employed. Therefore, we present a taxonomy highlighting the various facets of deploying synthetic data for advanced analytics systems. Furthermore, we identify typical application scenarios for synthetic data to assess the current state of adoption and thereby unveil missed opportunities to pave the way for further research.
       


### [Online Bayesian prediction of remaining useful life for gamma degradation process under conjugate priors](https://arxiv.org/abs/2212.02688)

**Author:**
Ancha Xu

**Abstract:**
Gamma process has been extensively used to model monotone degradation data. Statistical inference for the gamma process is difficult due to the complex parameter structure involved in the likelihood function. In this paper, we derive a conjugate prior for the homogeneous gamma process, and some properties of the prior distribution are explored. Three algorithms (Gibbs sampling, discrete grid sampling, and sampling importance resampling) are well designed to generate posterior samples of the model parameters, which can greatly lessen the challenge of posterior inference. Simulation studies show that the proposed algorithms have high computational efficiency and estimation precision. The conjugate prior is then extended to the case of the gamma process with heterogeneous effects. With this conjugate structure, the posterior distribution of the parameters can be updated recursively, and an efficient online algorithm is developed to predict remaining useful life of multiple systems. The effectiveness of the proposed online algorithm is illustrated by two real cases.
       


### [Digital Twin for Real-time Li-ion Battery State of Health Estimation with Partially Discharged Cycling Data](https://arxiv.org/abs/2212.04622)

**Authors:**
Yan Qin, Anushiya Arunan, Chau Yuen

**Abstract:**
To meet the fairly high safety and reliability requirements in practice, the state of health (SOH) estimation of Lithium-ion batteries (LIBs), which has a close relationship with the degradation performance, has been extensively studied with the widespread applications of various electronics. The conventional SOH estimation approaches with digital twin are end-of-cycle estimation that require the completion of a full charge/discharge cycle to observe the maximum available capacity. However, under dynamic operating conditions with partially discharged data, it is impossible to sense accurate real-time SOH estimation for LIBs. To bridge this research gap, we put forward a digital twin framework to gain the capability of sensing the battery's SOH on the fly, updating the physical battery model. The proposed digital twin solution consists of three core components to enable real-time SOH estimation without requiring a complete discharge. First, to handle the variable training cycling data, the energy discrepancy-aware cycling synchronization is proposed to align cycling data with guaranteeing the same data structure. Second, to explore the temporal importance of different training sampling times, a time-attention SOH estimation model is developed with data encoding to capture the degradation behavior over cycles, excluding adverse influences of unimportant samples. Finally, for online implementation, a similarity analysis-based data reconstruction has been put forward to provide real-time SOH estimation without requiring a full discharge cycle. Through a series of results conducted on a widely used benchmark, the proposed method yields the real-time SOH estimation with errors less than 1% for most sampling times in ongoing cycles.
       


### [Acela: Predictable Datacenter-level Maintenance Job Scheduling](https://arxiv.org/abs/2212.05155)

**Authors:**
Yi Ding, Aijia Gao, Thibaud Ryden, Kaushik Mitra, Sukumar Kalmanje, Yanai Golany, Michael Carbin, Henry Hoffmann

**Abstract:**
Datacenter operators ensure fair and regular server maintenance by using automated processes to schedule maintenance jobs to complete within a strict time budget. Automating this scheduling problem is challenging because maintenance job duration varies based on both job type and hardware. While it is tempting to use prior machine learning techniques for predicting job duration, we find that the structure of the maintenance job scheduling problem creates a unique challenge. In particular, we show that prior machine learning methods that produce the lowest error predictions do not produce the best scheduling outcomes due to asymmetric costs. Specifically, underpredicting maintenance job duration has results in more servers being taken offline and longer server downtime than overpredicting maintenance job duration. The system cost of underprediction is much larger than that of overprediction.
  We present Acela, a machine learning system for predicting maintenance job duration, which uses quantile regression to bias duration predictions toward overprediction. We integrate Acela into a maintenance job scheduler and evaluate it on datasets from large-scale, production datacenters. Compared to machine learning based predictors from prior work, Acela reduces the number of servers that are taken offline by 1.87-4.28X, and reduces the server offline time by 1.40-2.80X.
       


### [Multi-Dimensional Self Attention based Approach for Remaining Useful Life Estimation](https://arxiv.org/abs/2212.05772)

**Authors:**
Zhi Lai, Mengjuan Liu, Yunzhu Pan, Dajiang Chen

**Abstract:**
Remaining Useful Life (RUL) estimation plays a critical role in Prognostics and Health Management (PHM). Traditional machine health maintenance systems are often costly, requiring sufficient prior expertise, and are difficult to fit into highly complex and changing industrial scenarios. With the widespread deployment of sensors on industrial equipment, building the Industrial Internet of Things (IIoT) to interconnect these devices has become an inexorable trend in the development of the digital factory. Using the device's real-time operational data collected by IIoT to get the estimated RUL through the RUL prediction algorithm, the PHM system can develop proactive maintenance measures for the device, thus, reducing maintenance costs and decreasing failure times during operation. This paper carries out research into the remaining useful life prediction model for multi-sensor devices in the IIoT scenario. We investigated the mainstream RUL prediction models and summarized the basic steps of RUL prediction modeling in this scenario. On this basis, a data-driven approach for RUL estimation is proposed in this paper. It employs a Multi-Head Attention Mechanism to fuse the multi-dimensional time-series data output from multiple sensors, in which the attention on features is used to capture the interactions between features and attention on sequences is used to learn the weights of time steps. Then, the Long Short-Term Memory Network is applied to learn the features of time series. We evaluate the proposed model on two benchmark datasets (C-MAPSS and PHM08), and the results demonstrate that it outperforms the state-of-art models. Moreover, through the interpretability of the multi-head attention mechanism, the proposed model can provide a preliminary explanation of engine degradation. Therefore, this approach is promising for predictive maintenance in IIoT scenarios.
       


### [Agnostic Learning for Packing Machine Stoppage Prediction in Smart Factories](https://arxiv.org/abs/2212.06288)

**Authors:**
Gabriel Filios, Ioannis Katsidimas, Sotiris Nikoletseas, Stefanos H. Panagiotou, Theofanis P. Raptis

**Abstract:**
The cyber-physical convergence is opening up new business opportunities for industrial operators. The need for deep integration of the cyber and the physical worlds establishes a rich business agenda towards consolidating new system and network engineering approaches. This revolution would not be possible without the rich and heterogeneous sources of data, as well as the ability of their intelligent exploitation, mainly due to the fact that data will serve as a fundamental resource to promote Industry 4.0. One of the most fruitful research and practice areas emerging from this data-rich, cyber-physical, smart factory environment is the data-driven process monitoring field, which applies machine learning methodologies to enable predictive maintenance applications. In this paper, we examine popular time series forecasting techniques as well as supervised machine learning algorithms in the applied context of Industry 4.0, by transforming and preprocessing the historical industrial dataset of a packing machine's operational state recordings (real data coming from the production line of a manufacturing plant from the food and beverage domain). In our methodology, we use only a single signal concerning the machine's operational status to make our predictions, without considering other operational variables or fault and warning signals, hence its characterization as ``agnostic''. In this respect, the results demonstrate that the adopted methods achieve a quite promising performance on three targeted use cases.
       


### [Real-time Health Monitoring of Heat Exchangers using Hypernetworks and PINNs](https://arxiv.org/abs/2212.10032)

**Authors:**
Ritam Majumdar, Vishal Jadhav, Anirudh Deodhar, Shirish Karande, Lovekesh Vig, Venkataramana Runkana

**Abstract:**
We demonstrate a Physics-informed Neural Network (PINN) based model for real-time health monitoring of a heat exchanger, that plays a critical role in improving energy efficiency of thermal power plants. A hypernetwork based approach is used to enable the domain-decomposed PINN learn the thermal behavior of the heat exchanger in response to dynamic boundary conditions, eliminating the need to re-train. As a result, we achieve orders of magnitude reduction in inference time in comparison to existing PINNs, while maintaining the accuracy on par with the physics-based simulations. This makes the approach very attractive for predictive maintenance of the heat exchanger in digital twin environments.
       


### [Failure type detection and predictive maintenance for the next generation of imaging atmospheric Cherenkov telescopes](https://arxiv.org/abs/2212.12381)

**Authors:**
Federico Incardona, Alessandro Costa, Kevin Munari

**Abstract:**
The next generation of imaging atmospheric Cherenkov telescopes will be composed of hundreds of telescopes working together to attempt to unveil some fundamental physics of the high-energy Universe. Along with the scientific data, a large volume of housekeeping and auxiliary data coming from weather stations, instrumental sensors, logging files, etc., will be collected as well. Driven by supervised and reinforcement learning algorithms, such data can be exploited for applying predictive maintenance and failure type detection to these astrophysical facilities. In this paper, we present the project aiming to trigger the development of a model that will be able to predict, just in time, forthcoming component failures along with their kind and severity
       


### [Tool flank wear prediction using high-frequency machine data from industrial edge device](https://arxiv.org/abs/2212.13905)

**Authors:**
D. Bilgili, G. Kecibas, C. Besirova, M. R. Chehrehzad, G. Burun, T. Pehlivan, U. Uresin, E. Emekli, I. Lazoglu

**Abstract:**
Tool flank wear monitoring can minimize machining downtime costs while increasing productivity and product quality. In some industrial applications, only a limited level of tool wear is allowed to attain necessary tolerances. It may become challenging to monitor a limited level of tool wear in the data collected from the machine due to the other components, such as the flexible vibrations of the machine, dominating the measurement signals. In this study, a tool wear monitoring technique to predict limited levels of tool wear from the spindle motor current and dynamometer measurements is presented. High-frequency spindle motor current data is collected with an industrial edge device while the cutting forces and torque are measured with a rotary dynamometer in drilling tests for a selected number of holes. Feature engineering is conducted to identify the statistical features of the measurement signals that are most sensitive to small changes in tool wear. A neural network based on the long short-term memory (LSTM) architecture is developed to predict tool flank wear from the measured spindle motor current and dynamometer signals. It is demonstrated that the proposed technique predicts tool flank wear with good accuracy and high computational efficiency. The proposed technique can easily be implemented in an industrial edge device as a real-time predictive maintenance application to minimize the costs due to manufacturing downtime and tool underuse or overuse.
       


### [Similarity-Based Predictive Maintenance Framework for Rotating Machinery](https://arxiv.org/abs/2212.14550)

**Authors:**
Sulaiman Aburakhia, Tareq Tayeh, Ryan Myers, Abdallah Shami

**Abstract:**
Within smart manufacturing, data driven techniques are commonly adopted for condition monitoring and fault diagnosis of rotating machinery. Classical approaches use supervised learning where a classifier is trained on labeled data to predict or classify different operational states of the machine. However, in most industrial applications, labeled data is limited in terms of its size and type. Hence, it cannot serve the training purpose. In this paper, this problem is tackled by addressing the classification task as a similarity measure to a reference sample rather than a supervised classification task. Similarity-based approaches require a limited amount of labeled data and hence, meet the requirements of real-world industrial applications. Accordingly, the paper introduces a similarity-based framework for predictive maintenance (PdM) of rotating machinery. For each operational state of the machine, a reference vibration signal is generated and labeled according to the machine's operational condition. Consequentially, statistical time analysis, fast Fourier transform (FFT), and short-time Fourier transform (STFT) are used to extract features from the captured vibration signals. For each feature type, three similarity metrics, namely structural similarity measure (SSM), cosine similarity, and Euclidean distance are used to measure the similarity between test signals and reference signals in the feature space. Hence, nine settings in terms of feature type-similarity measure combinations are evaluated. Experimental results confirm the effectiveness of similarity-based approaches in achieving very high accuracy with moderate computational requirements compared to machine learning (ML)-based methods. Further, the results indicate that using FFT features with cosine similarity would lead to better performance compared to the other settings.
       


# 2023
## January
### [Bayesian Weapon System Reliability Modeling with Cox-Weibull Neural Network](https://arxiv.org/abs/2301.01850)

**Authors:**
Michael Potter, Benny Cheng

**Abstract:**
We propose to integrate weapon system features (such as weapon system manufacturer, deployment time and location, storage time and location, etc.) into a parameterized Cox-Weibull [1] reliability model via a neural network, like DeepSurv [2], to improve predictive maintenance. In parallel, we develop an alternative Bayesian model by parameterizing the Weibull parameters with a neural network and employing dropout methods such as Monte-Carlo (MC)-dropout for comparative purposes. Due to data collection procedures in weapon system testing we employ a novel interval-censored log-likelihood which incorporates Monte-Carlo Markov Chain (MCMC) [3] sampling of the Weibull parameters during gradient descent optimization. We compare classification metrics such as receiver operator curve (ROC) area under the curve (AUC), precision-recall (PR) AUC, and F scores to show our model generally outperforms traditional powerful models such as XGBoost and the current standard conditional Weibull probability density estimation model.
       


### [Interaction models for remaining useful life estimation](https://arxiv.org/abs/2301.05029)

**Authors:**
Dmitry Zhevnenko, Mikhail Kazantsev, Ilya Makarov

**Abstract:**
The paper deals with the problem of controlling the state of industrial devices according to the readings of their sensors. The current methods rely on one approach to feature extraction in which the prediction occurs. We proposed a technique to build a scalable model that combines multiple different feature extractor blocks. A new model based on sequential sensor space analysis achieves state-of-the-art results on the C-MAPSS benchmark for equipment remaining useful life estimation. The resulting model performance was validated including the prediction changes with scaling.
       


### [Explicit Context Integrated Recurrent Neural Network for Sensor Data Applications](https://arxiv.org/abs/2301.05031)

**Authors:**
Rashmi Dutta Baruah, Mario Muñoz Organero

**Abstract:**
The development and progress in sensor, communication and computing technologies have led to data rich environments. In such environments, data can easily be acquired not only from the monitored entities but also from the surroundings where the entity is operating. The additional data that are available from the problem domain, which cannot be used independently for learning models, constitute context. Such context, if taken into account while learning, can potentially improve the performance of predictive models. Typically, the data from various sensors are present in the form of time series. Recurrent Neural Networks (RNNs) are preferred for such data as it can inherently handle temporal context. However, the conventional RNN models such as Elman RNN, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) in their present form do not provide any mechanism to integrate explicit contexts. In this paper, we propose a Context Integrated RNN (CiRNN) that enables integrating explicit contexts represented in the form of contextual features. In CiRNN, the network weights are influenced by contextual features in such a way that the primary input features which are more relevant to a given context are given more importance. To show the efficacy of CiRNN, we selected an application domain, engine health prognostics, which captures data from various sensors and where contextual information is available. We used the NASA Turbofan Engine Degradation Simulation dataset for estimating Remaining Useful Life (RUL) as it provides contextual information. We compared CiRNN with baseline models as well as the state-of-the-art methods. The experimental results show an improvement of 39% and 87% respectively, over state-of-the art models, when performance is measured with RMSE and score from an asymmetric scoring function. The latter measure is specific to the task of RUL estimation.
       


### [Explainable, Interpretable & Trustworthy AI for Intelligent Digital Twin: Case Study on Remaining Useful Life](https://arxiv.org/abs/2301.06676)

**Authors:**
Kazuma Kobayashi, Syed Bahauddin Alam

**Abstract:**
Artificial intelligence (AI) and Machine learning (ML) are increasingly used in energy and engineering systems, but these models must be fair, unbiased, and explainable. It is critical to have confidence in AI's trustworthiness. ML techniques have been useful in predicting important parameters and in improving model performance. However, for these AI techniques to be useful for making decisions, they need to be audited, accounted for, and easy to understand. Therefore, the use of explainable AI (XAI) and interpretable machine learning (IML) is crucial for the accurate prediction of prognostics, such as remaining useful life (RUL), in a digital twin system, to make it intelligent while ensuring that the AI model is transparent in its decision-making processes and that the predictions it generates can be understood and trusted by users. By using AI that is explainable, interpretable, and trustworthy, intelligent digital twin systems can make more accurate predictions of RUL, leading to better maintenance and repair planning, and ultimately, improved system performance. The objective of this paper is to explain the ideas of XAI and IML and to justify the important role of AI/ML in the digital twin framework and components, which requires XAI to understand the prediction better. This paper explains the importance of XAI and IML in both local and global aspects to ensure the use of trustworthy AI/ML applications for RUL prediction. We used the RUL prediction for the XAI and IML studies and leveraged the integrated Python toolbox for interpretable machine learning~(PiML).
       


### [Self-Supervised Learning for Data Scarcity in a Fatigue Damage Prognostic Problem](https://arxiv.org/abs/2301.08441)

**Authors:**
Anass Akrim, Christian Gogu, Rob Vingerhoeds, Michel Salaün

**Abstract:**
With the increasing availability of data for Prognostics and Health Management (PHM), Deep Learning (DL) techniques are now the subject of considerable attention for this application, often achieving more accurate Remaining Useful Life (RUL) predictions. However, one of the major challenges for DL techniques resides in the difficulty of obtaining large amounts of labelled data on industrial systems. To overcome this lack of labelled data, an emerging learning technique is considered in our work: Self-Supervised Learning, a sub-category of unsupervised learning approaches. This paper aims to investigate whether pre-training DL models in a self-supervised way on unlabelled sensors data can be useful for RUL estimation with only Few-Shots Learning, i.e. with scarce labelled data. In this research, a fatigue damage prognostics problem is addressed, through the estimation of the RUL of aluminum alloy panels (typical of aerospace structures) subject to fatigue cracks from strain gauge data. Synthetic datasets composed of strain data are used allowing to extensively investigate the influence of the dataset size on the predictive performance. Results show that the self-supervised pre-trained models are able to significantly outperform the non-pre-trained models in downstream RUL prediction task, and with less computational expense, showing promising results in prognostic tasks when only limited labelled data is available.
       


### [Developing Hybrid Machine Learning Models to Assign Health Score to Railcar Fleets for Optimal Decision Making](https://arxiv.org/abs/2301.08877)

**Authors:**
Mahyar Ejlali, Ebrahim Arian, Sajjad Taghiyeh, Kristina Chambers, Amir Hossein Sadeghi, Demet Cakdi, Robert B Handfield

**Abstract:**
A large amount of data is generated during the operation of a railcar fleet, which can easily lead to dimensional disaster and reduce the resiliency of the railcar network. To solve these issues and offer predictive maintenance, this research introduces a hybrid fault diagnosis expert system method that combines density-based spatial clustering of applications with noise (DBSCAN) and principal component analysis (PCA). Firstly, the DBSCAN method is used to cluster categorical data that are similar to one another within the same group. Secondly, PCA algorithm is applied to reduce the dimensionality of the data and eliminate redundancy in order to improve the accuracy of fault diagnosis. Finally, we explain the engineered features and evaluate the selected models by using the Gain Chart and Area Under Curve (AUC) metrics. We use the hybrid expert system model to enhance maintenance planning decisions by assigning a health score to the railcar system of the North American Railcar Owner (NARO). According to the experimental results, our expert model can detect 96.4% of failures within 50% of the sample. This suggests that our method is effective at diagnosing failures in railcars fleet.
       


### [Digital Twins for Marine Operations: A Brief Review on Their Implementation](https://arxiv.org/abs/2301.09574)

**Authors:**
Federico Zocco, Hsueh-Cheng Wang, Mien Van

**Abstract:**
While the concept of a digital twin to support maritime operations is gaining attention for predictive maintenance, real-time monitoring, control, and overall process optimization, clarity on its implementation is missing in the literature. Therefore, in this review we show how different authors implemented their digital twins, discuss our findings, and finally give insights on future research directions.
       


### [DODEM: DOuble DEfense Mechanism Against Adversarial Attacks Towards Secure Industrial Internet of Things Analytics](https://arxiv.org/abs/2301.09740)

**Authors:**
Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstract:**
Industrial Internet of Things (I-IoT) is a collaboration of devices, sensors, and networking equipment to monitor and collect data from industrial operations. Machine learning (ML) methods use this data to make high-level decisions with minimal human intervention. Data-driven predictive maintenance (PDM) is a crucial ML-based I-IoT application to find an optimal maintenance schedule for industrial assets. The performance of these ML methods can seriously be threatened by adversarial attacks where an adversary crafts perturbed data and sends it to the ML model to deteriorate its prediction performance. The models should be able to stay robust against these attacks where robustness is measured by how much perturbation in input data affects model performance. Hence, there is a need for effective defense mechanisms that can protect these models against adversarial attacks. In this work, we propose a double defense mechanism to detect and mitigate adversarial attacks in I-IoT environments. We first detect if there is an adversarial attack on a given sample using novelty detection algorithms. Then, based on the outcome of our algorithm, marking an instance as attack or normal, we select adversarial retraining or standard training to provide a secondary defense layer. If there is an attack, adversarial retraining provides a more robust model, while we apply standard training for regular samples. Since we may not know if an attack will take place, our adaptive mechanism allows us to consider irregular changes in data. The results show that our double defense strategy is highly efficient where we can improve model robustness by up to 64.6% and 52% compared to standard and adversarial retraining, respectively.
       


### [RobustPdM: Designing Robust Predictive Maintenance against Adversarial Attacks](https://arxiv.org/abs/2301.10822)

**Authors:**
Ayesha Siddique, Ripan Kumar Kundu, Gautam Raj Mode, Khaza Anuarul Hoque

**Abstract:**
The state-of-the-art predictive maintenance (PdM) techniques have shown great success in reducing maintenance costs and downtime of complicated machines while increasing overall productivity through extensive utilization of Internet-of-Things (IoT) and Deep Learning (DL). Unfortunately, IoT sensors and DL algorithms are both prone to cyber-attacks. For instance, DL algorithms are known for their susceptibility to adversarial examples. Such adversarial attacks are vastly under-explored in the PdM domain. This is because the adversarial attacks in the computer vision domain for classification tasks cannot be directly applied to the PdM domain for multivariate time series (MTS) regression tasks. In this work, we propose an end-to-end methodology to design adversarially robust PdM systems by extensively analyzing the effect of different types of adversarial attacks and proposing a novel adversarial defense technique for DL-enabled PdM models. First, we propose novel MTS Projected Gradient Descent (PGD) and MTS PGD with random restarts (PGD_r) attacks. Then, we evaluate the impact of MTS PGD and PGD_r along with MTS Fast Gradient Sign Method (FGSM) and MTS Basic Iterative Method (BIM) on Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Convolutional Neural Network (CNN), and Bi-directional LSTM based PdM system. Our results using NASA's turbofan engine dataset show that adversarial attacks can cause a severe defect (up to 11X) in the RUL prediction, outperforming the effectiveness of the state-of-the-art PdM attacks by 3X. Furthermore, we present a novel approximate adversarial training method to defend against adversarial attacks. We observe that approximate adversarial training can significantly improve the robustness of PdM models (up to 54X) and outperforms the state-of-the-art PdM defense methods by offering 3X more robustness.
       


### [Oscillating behavior of a compartmental model with retarded noisy dynamic infection rate](https://arxiv.org/abs/2301.12437)

**Authors:**
Michael Bestehorn, Thomas M. Michelitsch

**Abstract:**
Our study is based on an epidemiological compartmental model, the SIRS model. In the SIRS model, each individual is in one of the states susceptible (S), infected(I) or recovered (R), depending on its state of health. In compartment R, an individual is assumed to stay immune within a finite time interval only and then transfers back to the S compartment. We extend the model and allow for a feedback control of the infection rate by mitigation measures which are related to the number of infections. A finite response time of the feedback mechanism is supposed that changes the low-dimensional SIRS model into an infinite-dimensional set of integro-differential (delay-differential) equations. It turns out that the retarded feedback renders the originally stable endemic equilibrium of SIRS (stable focus) into an unstable focus if the delay exceeds a certain critical value. Nonlinear solutions show persistent regular oscillations of the number of infected and susceptible individuals. In the last part we include noise effects from the environment and allow for a fluctuating infection rate. This results in multiplicative noise terms and our model turns into a set of stochastic nonlinear integro-differential equations. Numerical solutions reveal an irregular behavior of repeated disease outbreaks in the form of infection waves with a variety of frequencies and amplitudes.
       


### [Stochastic Wasserstein Gradient Flows using Streaming Data with an Application in Predictive Maintenance](https://arxiv.org/abs/2301.12461)

**Authors:**
Nicolas Lanzetti, Efe C. Balta, Dominic Liao-McPherson, Florian Dörfler

**Abstract:**
We study estimation problems in safety-critical applications with streaming data. Since estimation problems can be posed as optimization problems in the probability space, we devise a stochastic projected Wasserstein gradient flow that keeps track of the belief of the estimated quantity and can consume samples from online data. We show the convergence properties of our algorithm. Our analysis combines recent advances in the Wasserstein space and its differential structure with more classical stochastic gradient descent. We apply our methodology for predictive maintenance of safety-critical processes: Our approach is shown to lead to superior performance when compared to classical least squares, enabling, among others, improved robustness for decision-making.
       


### [Continual Learning for Predictive Maintenance: Overview and Challenges](https://arxiv.org/abs/2301.12467)

**Authors:**
Julio Hurtado, Dario Salvati, Rudy Semola, Mattia Bosio, Vincenzo Lomonaco

**Abstract:**
Deep learning techniques have become one of the main propellers for solving engineering problems effectively and efficiently. For instance, Predictive Maintenance methods have been used to improve predictions of when maintenance is needed on different machines and operative contexts. However, deep learning methods are not without limitations, as these models are normally trained on a fixed distribution that only reflects the current state of the problem. Due to internal or external factors, the state of the problem can change, and the performance decreases due to the lack of generalization and adaptation. Contrary to this stationary training set, real-world applications change their environments constantly, creating the need to constantly adapt the model to evolving scenarios. To aid in this endeavor, Continual Learning methods propose ways to constantly adapt prediction models and incorporate new knowledge after deployment. Despite the advantages of these techniques, there are still challenges to applying them to real-world problems. In this work, we present a brief introduction to predictive maintenance, non-stationary environments, and continual learning, together with an extensive review of the current state of applying continual learning in real-world applications and specifically in predictive maintenance. We then discuss the current challenges of both predictive maintenance and continual learning, proposing future directions at the intersection of both areas. Finally, we propose a novel way to create benchmarks that favor the application of continuous learning methods in more realistic environments, giving specific examples of predictive maintenance.
       


## February
### [Energy-Based Survival Models for Predictive Maintenance](https://arxiv.org/abs/2302.00629)

**Authors:**
Olov Holmer, Erik Frisk, Mattias Krysander

**Abstract:**
Predictive maintenance is an effective tool for reducing maintenance costs. Its effectiveness relies heavily on the ability to predict the future state of health of the system, and for this survival models have shown to be very useful. Due to the complex behavior of system degradation, data-driven methods are often preferred, and neural network-based methods have been shown to perform particularly very well. Many neural network-based methods have been proposed and successfully applied to many problems. However, most models rely on assumptions that often are quite restrictive and there is an interest to find more expressive models. Energy-based models are promising candidates for this due to their successful use in other applications, which include natural language processing and computer vision. The focus of this work is therefore to investigate how energy-based models can be used for survival modeling and predictive maintenance. A key step in using energy-based models for survival modeling is the introduction of right-censored data, which, based on a maximum likelihood approach, is shown to be a straightforward process. Another important part of the model is the evaluation of the integral used to normalize the modeled probability density function, and it is shown how this can be done efficiently. The energy-based survival model is evaluated using both simulated data and experimental data in the form of starter battery failures from a fleet of vehicles, and its performance is found to be highly competitive compared to existing models.
       


### [Domain Adaptation via Alignment of Operation Profile for Remaining Useful Lifetime Prediction](https://arxiv.org/abs/2302.01704)

**Authors:**
Ismail Nejjar, Fabian Geissmann, Mengjie Zhao, Cees Taal, Olga Fink

**Abstract:**
Effective Prognostics and Health Management (PHM) relies on accurate prediction of the Remaining Useful Life (RUL). Data-driven RUL prediction techniques rely heavily on the representativeness of the available time-to-failure trajectories. Therefore, these methods may not perform well when applied to data from new units of a fleet that follow different operating conditions than those they were trained on. This is also known as domain shifts. Domain adaptation (DA) methods aim to address the domain shift problem by extracting domain invariant features. However, DA methods do not distinguish between the different phases of operation, such as steady states or transient phases. This can result in misalignment due to under- or over-representation of different operation phases. This paper proposes two novel DA approaches for RUL prediction based on an adversarial domain adaptation framework that considers the different phases of the operation profiles separately. The proposed methodologies align the marginal distributions of each phase of the operation profile in the source domain with its counterpart in the target domain. The effectiveness of the proposed methods is evaluated using the New Commercial Modular Aero-Propulsion System (N-CMAPSS) dataset, where sub-fleets of turbofan engines operating in one of the three different flight classes (short, medium, and long) are treated as separate domains. The experimental results show that the proposed methods improve the accuracy of RUL predictions compared to current state-of-the-art DA methods.
       


### [Fatigue monitoring and maneuver identification for vehicle fleets using the scattering transform](https://arxiv.org/abs/2302.02737)

**Authors:**
Leonhard Heindel, Peter Hantschke, Markus Kästner

**Abstract:**
Extensive monitoring comes at a prohibitive cost, limiting Predictive Maintenance strategies for vehicle fleets. This paper presents a measurement-based soft sensing technique where local strain gauges are only required for few reference vehicles, while the remaining fleet relies exclusively on accelerometers. The scattering transform is used to perform feature extraction, while principal component analysis provides a reduced, low dimensional data representation. This enables direct fatigue damage regression, parameterized from unlabeled usage data. Identification measurements allow for a physical interpretation of the reduced representation. The approach is demonstrated using experimental data from a sensor equipped eBike.
       


### [A Benchmark on Uncertainty Quantification for Deep Learning Prognostics](https://arxiv.org/abs/2302.04730)

**Authors:**
Luis Basora, Arthur Viens, Manuel Arias Chao, Xavier Olive

**Abstract:**
Reliable uncertainty quantification on RUL prediction is crucial for informative decision-making in predictive maintenance. In this context, we assess some of the latest developments in the field of uncertainty quantification for prognostics deep learning. This includes the state-of-the-art variational inference algorithms for Bayesian neural networks (BNN) as well as popular alternatives such as Monte Carlo Dropout (MCD), deep ensembles (DE) and heteroscedastic neural networks (HNN). All the inference techniques share the same inception deep learning architecture as a functional model. We performed hyperparameter search to optimize the main variational and learning parameters of the algorithms. The performance of the methods is evaluated on a subset of the large NASA NCMAPSS dataset for aircraft engines. The assessment includes RUL prediction accuracy, the quality of predictive uncertainty, and the possibility to break down the total predictive uncertainty into its aleatoric and epistemic parts. The results show no method clearly outperforms the others in all the situations. Although all methods are close in terms of accuracy, we find differences in the way they estimate uncertainty. Thus, DE and MCD generally provide more conservative predictive uncertainty than BNN. Surprisingly, HNN can achieve strong results without the added training complexity and extra parameters of the BNN. For tasks like active learning where a separation of epistemic and aleatoric uncertainty is required, radial BNN and MCD seem the best options.
       


### [A Lifetime Extended Energy Management Strategy for Fuel Cell Hybrid Electric Vehicles via Self-Learning Fuzzy Reinforcement Learning](https://arxiv.org/abs/2302.06236)

**Authors:**
Liang Guo, Zhongliang Li, Rachid Outbib

**Abstract:**
Modeling difficulty, time-varying model, and uncertain external inputs are the main challenges for energy management of fuel cell hybrid electric vehicles. In the paper, a fuzzy reinforcement learning-based energy management strategy for fuel cell hybrid electric vehicles is proposed to reduce fuel consumption, maintain the batteries' long-term operation, and extend the lifetime of the fuel cells system. Fuzzy Q-learning is a model-free reinforcement learning that can learn itself by interacting with the environment, so there is no need for modeling the fuel cells system. In addition, frequent startup of the fuel cells will reduce the remaining useful life of the fuel cells system. The proposed method suppresses frequent fuel cells startup by considering the penalty for the times of fuel cell startups in the reward of reinforcement learning. Moreover, applying fuzzy logic to approximate the value function in Q-Learning can solve continuous state and action space problems. Finally, a python-based training and testing platform verify the effectiveness and self-learning improvement of the proposed method under conditions of initial state change, model change and driving condition change.
       


### [STB-VMM: Swin Transformer Based Video Motion Magnification](https://arxiv.org/abs/2302.10001)

**Authors:**
Ricard Lado-Roigé, Marco A. Pérez

**Abstract:**
The goal of video motion magnification techniques is to magnify small motions in a video to reveal previously invisible or unseen movement. Its uses extend from bio-medical applications and deepfake detection to structural modal analysis and predictive maintenance. However, discerning small motion from noise is a complex task, especially when attempting to magnify very subtle, often sub-pixel movement. As a result, motion magnification techniques generally suffer from noisy and blurry outputs. This work presents a new state-of-the-art model based on the Swin Transformer, which offers better tolerance to noisy inputs as well as higher-quality outputs that exhibit less noise, blurriness, and artifacts than prior-art. Improvements in output image quality will enable more precise measurements for any application reliant on magnified video sequences, and may enable further development of video motion magnification techniques in new technical fields.
       


### [Impact of Thermal Variability on SOC Estimation Algorithms](https://arxiv.org/abs/2302.12978)

**Authors:**
Wasiue Ahmed, Mokhi Maan Siddiqui, Faheemullah Shaikh

**Abstract:**
While the efficiency of renewable energy components like inverters and PV panels is at an all-time high, there are still research gaps for batteries. Lithium-ion batteries have a lot of potential, but there are still some problems that need fixing, such as thermal management. Because of this, the battery management system accomplishes its goal. In order for a battery management system (BMS) to function properly, it must make accurate estimates of all relevant parameters, including state of health, state of charge, and temperature; however, for the purposes of this article, we will only discuss SOC. The goal of this article is to estimate the SOC of a lithium-ion battery at different temperatures. Comparing the Extended Kalam filter algorithm to coulomb counting at various temperatures concludes this exhaustive investigation. The graphene battery has the highest SOC when operated at the optimal temperature, as determined by extensive analysis and correlation between SOC and temperature is not linear
       


## March
### [Rule-based Out-Of-Distribution Detection](https://arxiv.org/abs/2303.01860)

**Authors:**
Giacomo De Bernardi, Sara Narteni, Enrico Cambiaso, Maurizio Mongelli

**Abstract:**
Out-of-distribution detection is one of the most critical issue in the deployment of machine learning. The data analyst must assure that data in operation should be compliant with the training phase as well as understand if the environment has changed in a way that autonomous decisions would not be safe anymore. The method of the paper is based on eXplainable Artificial Intelligence (XAI); it takes into account different metrics to identify any resemblance between in-distribution and out of, as seen by the XAI model. The approach is non-parametric and distributional assumption free. The validation over complex scenarios (predictive maintenance, vehicle platooning, covert channels in cybersecurity) corroborates both precision in detection and evaluation of training-operation conditions proximity. Results are available via open source and open data at the following link: https://github.com/giacomo97cnr/Rule-based-ODD.
       


### [On the Soundness of XAI in Prognostics and Health Management (PHM)](https://arxiv.org/abs/2303.05517)

**Authors:**
David Solís-Martín, Juan Galán-Páez, Joaquín Borrego-Díaz

**Abstract:**
The aim of Predictive Maintenance, within the field of Prognostics and Health Management (PHM), is to identify and anticipate potential issues in the equipment before these become critical. The main challenge to be addressed is to assess the amount of time a piece of equipment will function effectively before it fails, which is known as Remaining Useful Life (RUL). Deep Learning (DL) models, such as Deep Convolutional Neural Networks (DCNN) and Long Short-Term Memory (LSTM) networks, have been widely adopted to address the task, with great success. However, it is well known that this kind of black box models are opaque decision systems, and it may be hard to explain its outputs to stakeholders (experts in the industrial equipment). Due to the large number of parameters that determine the behavior of these complex models, understanding the reasoning behind the predictions is challenging. This work presents a critical and comparative revision on a number of XAI methods applied on time series regression model for PM. The aim is to explore XAI methods within time series regression, which have been less studied than those for time series classification. The model used during the experimentation is a DCNN trained to predict the RUL of an aircraft engine. The methods are reviewed and compared using a set of metrics that quantifies a number of desirable properties that any XAI method should fulfill. The results show that GRAD-CAM is the most robust method, and that the best layer is not the bottom one, as is commonly seen within the context of Image Processing.
       


### [Large-scale End-of-Life Prediction of Hard Disks in Distributed Datacenters](https://arxiv.org/abs/2303.08955)

**Authors:**
Rohan Mohapatra, Austin Coursey, Saptarshi Sengupta

**Abstract:**
On a daily basis, data centers process huge volumes of data backed by the proliferation of inexpensive hard disks. Data stored in these disks serve a range of critical functional needs from financial, and healthcare to aerospace. As such, premature disk failure and consequent loss of data can be catastrophic. To mitigate the risk of failures, cloud storage providers perform condition-based monitoring and replace hard disks before they fail. By estimating the remaining useful life of hard disk drives, one can predict the time-to-failure of a particular device and replace it at the right time, ensuring maximum utilization whilst reducing operational costs. In this work, large-scale predictive analyses are performed using severely skewed health statistics data by incorporating customized feature engineering and a suite of sequence learners. Past work suggests using LSTMs as an excellent approach to predicting remaining useful life. To this end, we present an encoder-decoder LSTM model where the context gained from understanding health statistics sequences aid in predicting an output sequence of the number of days remaining before a disk potentially fails. The models developed in this work are trained and tested across an exhaustive set of all of the 10 years of S.M.A.R.T. health data in circulation from Backblaze and on a wide variety of disk instances. It closes the knowledge gap on what full-scale training achieves on thousands of devices and advances the state-of-the-art by providing tangible metrics for evaluation and generalization for practitioners looking to extend their workflow to all years of health data in circulation across disk manufacturers. The encoder-decoder LSTM posted an RMSE of 0.83 during training and 0.86 during testing over the exhaustive 10 year data while being able to generalize competitively over other drives from the Seagate family.
       


### [Gate Recurrent Unit Network based on Hilbert-Schmidt Independence Criterion for State-of-Health Estimation](https://arxiv.org/abs/2303.09497)

**Authors:**
Ziyue Huang, Lujuan Dang, Yuqing Xie, Wentao Ma, Badong Chen

**Abstract:**
State-of-health (SOH) estimation is a key step in ensuring the safe and reliable operation of batteries. Due to issues such as varying data distribution and sequence length in different cycles, most existing methods require health feature extraction technique, which can be time-consuming and labor-intensive. GRU can well solve this problem due to the simple structure and superior performance, receiving widespread attentions. However, redundant information still exists within the network and impacts the accuracy of SOH estimation. To address this issue, a new GRU network based on Hilbert-Schmidt Independence Criterion (GRU-HSIC) is proposed. First, a zero masking network is used to transform all battery data measured with varying lengths every cycle into sequences of the same length, while still retaining information about the original data size in each cycle. Second, the Hilbert-Schmidt Independence Criterion (HSIC) bottleneck, which evolved from Information Bottleneck (IB) theory, is extended to GRU to compress the information from hidden layers. To evaluate the proposed method, we conducted experiments on datasets from the Center for Advanced Life Cycle Engineering (CALCE) of the University of Maryland and NASA Ames Prognostics Center of Excellence. Experimental results demonstrate that our model achieves higher accuracy than other recurrent models.
       


### [Fault Prognosis of Turbofan Engines: Eventual Failure Prediction and Remaining Useful Life Estimation](https://arxiv.org/abs/2303.12982)

**Authors:**
Joseph Cohen, Xun Huan, Jun Ni

**Abstract:**
In the era of industrial big data, prognostics and health management is essential to improve the prediction of future failures to minimize inventory, maintenance, and human costs. Used for the 2021 PHM Data Challenge, the new Commercial Modular Aero-Propulsion System Simulation dataset from NASA is an open-source benchmark containing simulated turbofan engine units flown under realistic flight conditions. Deep learning approaches implemented previously for this application attempt to predict the remaining useful life of the engine units, but have not utilized labeled failure mode information, impeding practical usage and explainability. To address these limitations, a new prognostics approach is formulated with a customized loss function to simultaneously predict the current health state, the eventual failing component(s), and the remaining useful life. The proposed method incorporates principal component analysis to orthogonalize statistical time-domain features, which are inputs into supervised regressors such as random forests, extreme random forests, XGBoost, and artificial neural networks. The highest performing algorithm, ANN-Flux, achieves AUROC and AUPR scores exceeding 0.95 for each classification. In addition, ANN-Flux reduces the remaining useful life RMSE by 38% for the same test split of the dataset compared to past work, with significantly less computational cost.
       


## April
### [A Self-attention Knowledge Domain Adaptation Network for Commercial Lithium-ion Batteries State-of-health Estimation under Shallow Cycles](https://arxiv.org/abs/2304.05084)

**Authors:**
Xin Chen, Yuwen Qin, Weidong Zhao, Qiming Yang, Ningbo Cai, Kai Wu

**Abstract:**
Accurate state-of-health (SOH) estimation is critical to guarantee the safety, efficiency and reliability of battery-powered applications. Most SOH estimation methods focus on the 0-100\% full state-of-charge (SOC) range that has similar distributions. However, the batteries in real-world applications usually work in the partial SOC range under shallow-cycle conditions and follow different degradation profiles with no labeled data available, thus making SOH estimation challenging. To estimate shallow-cycle battery SOH, a novel unsupervised deep transfer learning method is proposed to bridge different domains using self-attention distillation module and multi-kernel maximum mean discrepancy technique. The proposed method automatically extracts domain-variant features from charge curves to transfer knowledge from the large-scale labeled full cycles to the unlabeled shallow cycles. The CALCE and SNL battery datasets are employed to verify the effectiveness of the proposed method to estimate the battery SOH for different SOC ranges, temperatures, and discharge rates. The proposed method achieves a root-mean-square error within 2\% and outperforms other transfer learning methods for different SOC ranges. When applied to batteries with different operating conditions and from different manufacturers, the proposed method still exhibits superior SOH estimation performance. The proposed method is the first attempt at accurately estimating battery SOH under shallow-cycle conditions without needing a full-cycle characteristic test.
       


### [CyFormer: Accurate State-of-Health Prediction of Lithium-Ion Batteries via Cyclic Attention](https://arxiv.org/abs/2304.08502)

**Authors:**
Zhiqiang Nie, Jiankun Zhao, Qicheng Li, Yong Qin

**Abstract:**
Predicting the State-of-Health (SoH) of lithium-ion batteries is a fundamental task of battery management systems on electric vehicles. It aims at estimating future SoH based on historical aging data. Most existing deep learning methods rely on filter-based feature extractors (e.g., CNN or Kalman filters) and recurrent time sequence models. Though efficient, they generally ignore cyclic features and the domain gap between training and testing batteries. To address this problem, we present CyFormer, a transformer-based cyclic time sequence model for SoH prediction. Instead of the conventional CNN-RNN structure, we adopt an encoder-decoder architecture. In the encoder, row-wise and column-wise attention blocks effectively capture intra-cycle and inter-cycle connections and extract cyclic features. In the decoder, the SoH queries cross-attend to these features to form the final predictions. We further utilize a transfer learning strategy to narrow the domain gap between the training and testing set. To be specific, we use fine-tuning to shift the model to a target working condition. Finally, we made our model more efficient by pruning. The experiment shows that our method attains an MAE of 0.75\% with only 10\% data for fine-tuning on a testing battery, surpassing prior methods by a large margin. Effective and robust, our method provides a potential solution for all cyclic time sequence prediction tasks.
       


### [Federated Learning for Predictive Maintenance and Quality Inspection in Industrial Applications](https://arxiv.org/abs/2304.11101)

**Authors:**
Viktorija Pruckovskaja, Axel Weissenfeld, Clemens Heistracher, Anita Graser, Julia Kafka, Peter Leputsch, Daniel Schall, Jana Kemnitz

**Abstract:**
Data-driven machine learning is playing a crucial role in the advancements of Industry 4.0, specifically in enhancing predictive maintenance and quality inspection. Federated learning (FL) enables multiple participants to develop a machine learning model without compromising the privacy and confidentiality of their data. In this paper, we evaluate the performance of different FL aggregation methods and compare them to central and local training approaches. Our study is based on four datasets with varying data distributions. The results indicate that the performance of FL is highly dependent on the data and its distribution among clients. In some scenarios, FL can be an effective alternative to traditional central or local training methods. Additionally, we introduce a new federated learning dataset from a real-world quality inspection setting.
       


### [Controlled physics-informed data generation for deep learning-based remaining useful life prediction under unseen operation conditions](https://arxiv.org/abs/2304.11702)

**Authors:**
Jiawei Xiong, Olga Fink, Jian Zhou, Yizhong Ma

**Abstract:**
Limited availability of representative time-to-failure (TTF) trajectories either limits the performance of deep learning (DL)-based approaches on remaining useful life (RUL) prediction in practice or even precludes their application. Generating synthetic data that is physically plausible is a promising way to tackle this challenge. In this study, a novel hybrid framework combining the controlled physics-informed data generation approach with a deep learning-based prediction model for prognostics is proposed. In the proposed framework, a new controlled physics-informed generative adversarial network (CPI-GAN) is developed to generate synthetic degradation trajectories that are physically interpretable and diverse. Five basic physics constraints are proposed as the controllable settings in the generator. A physics-informed loss function with penalty is designed as the regularization term, which ensures that the changing trend of system health state recorded in the synthetic data is consistent with the underlying physical laws. Then, the generated synthetic data is used as input of the DL-based prediction model to obtain the RUL estimations. The proposed framework is evaluated based on new Commercial Modular Aero-Propulsion System Simulation (N-CMAPSS), a turbofan engine prognostics dataset where a limited avail-ability of TTF trajectories is assumed. The experimental results demonstrate that the proposed framework is able to generate synthetic TTF trajectories that are consistent with underlying degradation trends. The generated trajectories enable to significantly improve the accuracy of RUL predictions.
       


### [Learning battery model parameter dynamics from data with recursive Gaussian process regression](https://arxiv.org/abs/2304.13666)

**Authors:**
Antti Aitio, Dominik Jöst, Dirk Uwe Sauer, David A. Howey

**Abstract:**
Estimating state of health is a critical function of a battery management system but remains challenging due to the variability of operating conditions and usage requirements of real applications. As a result, techniques based on fitting equivalent circuit models may exhibit inaccuracy at extremes of performance and over long-term ageing, or instability of parameter estimates. Pure data-driven techniques, on the other hand, suffer from lack of generality beyond their training dataset. In this paper, we propose a hybrid approach combining data- and model-driven techniques for battery health estimation. Specifically, we demonstrate a Bayesian data-driven method, Gaussian process regression, to estimate model parameters as functions of states, operating conditions, and lifetime. Computational efficiency is ensured through a recursive approach yielding a unified joint state-parameter estimator that learns parameter dynamics from data and is robust to gaps and varying operating conditions. Results show the efficacy of the method, on both simulated and measured data, including accurate estimates and forecasts of battery capacity and internal resistance. This opens up new opportunities to understand battery ageing in real applications.
       


## May
### [Uncertainty Quantification in Machine Learning for Engineering Design and Health Prognostics: A Tutorial](https://arxiv.org/abs/2305.04933)

**Authors:**
Venkat Nemani, Luca Biggio, Xun Huan, Zhen Hu, Olga Fink, Anh Tran, Yan Wang, Xiaoge Zhang, Chao Hu

**Abstract:**
On top of machine learning models, uncertainty quantification (UQ) functions as an essential layer of safety assurance that could lead to more principled decision making by enabling sound risk assessment and management. The safety and reliability improvement of ML models empowered by UQ has the potential to significantly facilitate the broad adoption of ML solutions in high-stakes decision settings, such as healthcare, manufacturing, and aviation, to name a few. In this tutorial, we aim to provide a holistic lens on emerging UQ methods for ML models with a particular focus on neural networks and the applications of these UQ methods in tackling engineering design as well as prognostics and health management problems. Toward this goal, we start with a comprehensive classification of uncertainty types, sources, and causes pertaining to UQ of ML models. Next, we provide a tutorial-style description of several state-of-the-art UQ methods: Gaussian process regression, Bayesian neural network, neural network ensemble, and deterministic UQ methods focusing on spectral-normalized neural Gaussian process. Established upon the mathematical formulations, we subsequently examine the soundness of these UQ methods quantitatively and qualitatively (by a toy regression example) to examine their strengths and shortcomings from different dimensions. Then, we review quantitative metrics commonly used to assess the quality of predictive uncertainty in classification and regression problems. Afterward, we discuss the increasingly important role of UQ of ML models in solving challenging problems in engineering design and health prognostics. Two case studies with source codes available on GitHub are used to demonstrate these UQ methods and compare their performance in the life prediction of lithium-ion batteries at the early stage and the remaining useful life prediction of turbofan engines.
       


### [To transfer or not transfer: Unified transferability metric and analysis](https://arxiv.org/abs/2305.07741)

**Authors:**
Qianshan Zhan, Xiao-Jun Zeng

**Abstract:**
In transfer learning, transferability is one of the most fundamental problems, which aims to evaluate the effectiveness of arbitrary transfer tasks. Existing research focuses on classification tasks and neglects domain or task differences. More importantly, there is a lack of research to determine whether to transfer or not. To address these, we propose a new analytical approach and metric, Wasserstein Distance based Joint Estimation (WDJE), for transferability estimation and determination in a unified setting: classification and regression problems with domain and task differences. The WDJE facilitates decision-making on whether to transfer or not by comparing the target risk with and without transfer. To enable the comparison, we approximate the target transfer risk by proposing a non-symmetric, easy-to-understand and easy-to-calculate target risk bound that is workable even with limited target labels. The proposed bound relates the target risk to source model performance, domain and task differences based on Wasserstein distance. We also extend our bound into unsupervised settings and establish the generalization bound from finite empirical samples. Our experiments in image classification and remaining useful life regression prediction illustrate the effectiveness of the WDJE in determining whether to transfer or not, and the proposed bound in approximating the target transfer risk.
       


### [A Federated Learning-based Industrial Health Prognostics for Heterogeneous Edge Devices using Matched Feature Extraction](https://arxiv.org/abs/2305.07854)

**Authors:**
Anushiya Arunan, Yan Qin, Xiaoli Li, Chau Yuen

**Abstract:**
Data-driven industrial health prognostics require rich training data to develop accurate and reliable predictive models. However, stringent data privacy laws and the abundance of edge industrial data necessitate decentralized data utilization. Thus, the industrial health prognostics field is well suited to significantly benefit from federated learning (FL), a decentralized and privacy-preserving learning technique. However, FL-based health prognostics tasks have hardly been investigated due to the complexities of meaningfully aggregating model parameters trained from heterogeneous data to form a high performing federated model. Specifically, data heterogeneity among edge devices, stemming from dissimilar degradation mechanisms and unequal dataset sizes, poses a critical statistical challenge for developing accurate federated models. We propose a pioneering FL-based health prognostic model with a feature similarity-matched parameter aggregation algorithm to discriminatingly learn from heterogeneous edge data. The algorithm searches across the heterogeneous locally trained models and matches neurons with probabilistically similar feature extraction functions first, before selectively averaging them to form the federated model parameters. As the algorithm only averages similar neurons, as opposed to conventional naive averaging of coordinate-wise neurons, the distinct feature extractors of local models are carried over with less dilution to the resultant federated model. Using both cyclic degradation data of Li-ion batteries and non-cyclic data of turbofan engines, we demonstrate that the proposed method yields accuracy improvements as high as 44.5\% and 39.3\% for state-of-health estimation and remaining useful life estimation, respectively.
       


### [Sound-to-Vibration Transformation for Sensorless Motor Health Monitoring](https://arxiv.org/abs/2305.07960)

**Authors:**
Ozer Can Devecioglu, Serkan Kiranyaz, Amer Elhmes, Sadok Sassi, Turker Ince, Onur Avci, Mohammad Hesam Soleimani-Babakamali, Ertugrul Taciroglu, Moncef Gabbouj

**Abstract:**
Automatic sensor-based detection of motor failures such as bearing faults is crucial for predictive maintenance in various industries. Numerous methodologies have been developed over the years to detect bearing faults. Despite the appearance of numerous different approaches for diagnosing faults in motors have been proposed, vibration-based methods have become the de facto standard and the most commonly used techniques. However, acquiring reliable vibration signals, especially from rotating machinery, can sometimes be infeasibly difficult due to challenging installation and operational conditions (e.g., variations on accelerometer locations on the motor body), which will not only alter the signal patterns significantly but may also induce severe artifacts. Moreover, sensors are costly and require periodic maintenance to sustain a reliable signal acquisition. To address these drawbacks and void the need for vibration sensors, in this study, we propose a novel sound-to-vibration transformation method that can synthesize realistic vibration signals directly from the sound measurements regardless of the working conditions, fault type, and fault severity. As a result, using this transformation, the data acquired by a simple sound recorder, e.g., a mobile phone, can be transformed into the vibration signal, which can then be used for fault detection by a pre-trained model. The proposed method is extensively evaluated over the benchmark Qatar University Dual-Machine Bearing Fault Benchmark dataset (QU-DMBF), which encapsulates sound and vibration data from two different machines operating under various conditions. Experimental results show that this novel approach can synthesize such realistic vibration signals that can directly be used for reliable and highly accurate motor health monitoring.
       


### [Analyzing the Stance of Facebook Posts on Abortion Considering State-level Health and Social Compositions](https://arxiv.org/abs/2305.09889)

**Authors:**
Ana Aleksandric, Henry Isaac Anderson, Anisha Dangal, Gabriela Mustata Wilson, Shirin Nilizadeh

**Abstract:**
Abortion remains one of the most controversial topics, especially after overturning Roe v. Wade ruling in the United States. Previous literature showed that the illegality of abortion could have serious consequences, as women might seek unsafe pregnancy terminations leading to increased maternal mortality rates and negative effects on their reproductive health. Therefore, the stances of the abortion-related Facebook posts were analyzed at the state level in the United States from May 4 until June 30, 2022, right after the Supreme Court's decision was disclosed. In more detail, the pre-trained Transformer architecture-based model was fine-tuned on a manually labeled training set to obtain a stance detection model suitable for the collected dataset. Afterward, we employed appropriate statistical tests to examine the relationships between public opinion regarding abortion, abortion legality, political leaning, and factors measuring the overall population's health, health knowledge, and vulnerability per state. We found that states with a higher number of views against abortion also have higher infant and maternal mortality rates. Furthermore, the stance of social media posts per state is mostly matching with the current abortion laws in these states. While aligned with existing literature, these findings indicate how public opinion, laws, and women's and infants' health are related, and interventions are required to educate and protect women, especially in vulnerable populations.
       


### [A hybrid feature learning approach based on convolutional kernels for ATM fault prediction using event-log data](https://arxiv.org/abs/2305.10059)

**Authors:**
Víctor Manuel Vargas, Riccardo Rosati, César Hervás-Martínez, Adriano Mancini, Luca Romeo, Pedro Antonio Gutiérrez

**Abstract:**
Predictive Maintenance (PdM) methods aim to facilitate the scheduling of maintenance work before equipment failure. In this context, detecting early faults in automated teller machines (ATMs) has become increasingly important since these machines are susceptible to various types of unpredictable failures. ATMs track execution status by generating massive event-log data that collect system messages unrelated to the failure event. Predicting machine failure based on event logs poses additional challenges, mainly in extracting features that might represent sequences of events indicating impending failures. Accordingly, feature learning approaches are currently being used in PdM, where informative features are learned automatically from minimally processed sensor data. However, a gap remains to be seen on how these approaches can be exploited for deriving relevant features from event-log-based data. To fill this gap, we present a predictive model based on a convolutional kernel (MiniROCKET and HYDRA) to extract features from the original event-log data and a linear classifier to classify the sample based on the learned features. The proposed methodology is applied to a significant real-world collected dataset. Experimental results demonstrated how one of the proposed convolutional kernels (i.e. HYDRA) exhibited the best classification performance (accuracy of 0.759 and AUC of 0.693). In addition, statistical analysis revealed that the HYDRA and MiniROCKET models significantly overcome one of the established state-of-the-art approaches in time series classification (InceptionTime), and three non-temporal ML methods from the literature. The predictive model was integrated into a container-based decision support system to support operators in the timely maintenance of ATMs.
       


### [Estimation of Remaining Useful Life and SOH of Lithium Ion Batteries (For EV Vehicles)](https://arxiv.org/abs/2305.10298)

**Author:**
Ganesh Kumar

**Abstract:**
Lithium-ion batteries are widely used in various applications, including portable electronic devices, electric vehicles, and renewable energy storage systems. Accurately estimating the remaining useful life of these batteries is crucial for ensuring their optimal performance, preventing unexpected failures, and reducing maintenance costs. In this paper, we present a comprehensive review of the existing approaches for estimating the remaining useful life of lithium-ion batteries, including data-driven methods, physics-based models, and hybrid approaches. We also propose a novel approach based on machine learning techniques for accurately predicting the remaining useful life of lithium-ion batteries. Our approach utilizes various battery performance parameters, including voltage, current, and temperature, to train a predictive model that can accurately estimate the remaining useful life of the battery. We evaluate the performance of our approach on a dataset of lithium-ion battery cycles and compare it with other state-of-the-art methods. The results demonstrate the effectiveness of our proposed approach in accurately estimating the remaining useful life of lithium-ion batteries.
       


### [Frequency domain parametric estimation of fractional order impedance models for Li-ion batteries](https://arxiv.org/abs/2305.15840)

**Authors:**
Freja Vandeputte, Noël Hallemans, Jishnu Ayyangatu Kuzhiyil, Nessa Fereshteh Saniee, Widanalage Dhammika Widanage, John Lataire

**Abstract:**
The impedance of a Li-ion battery contains information about its state of charge (SOC), state of health (SOH) and remaining useful life (RUL). Commonly, electrochemical impedance spectroscopy (EIS) is used as a nonparametric data-driven technique for estimating this impedance from current and voltage measurements. In this article, however, we propose a consistent parametric estimation method based on a fractional order equivalent circuit model (ECM) of the battery impedance. Contrary to the nonparametric impedance estimate, which is only defined at the discrete set of excited frequencies, the parametric estimate can be evaluated in every frequency of the frequency band of interest. Moreover, we are not limited to a single sine or multisine excitation signal. Instead, any persistently exciting signal, like for example a noise excitation signal, will suffice. The parametric estimation method is first validated on simulations and then applied to measurements of commercial Samsung 48X cells. For now, only batteries in rest, i.e. at a constant SOC after relaxation, are considered.
       


## June
### [A metric for assessing and optimizing data-driven prognostic algorithms for predictive maintenance](https://arxiv.org/abs/2306.03759)

**Authors:**
Antonios Kamariotis, Konstantinos Tatsis, Eleni Chatzi, Kai Goebel, Daniel Straub

**Abstract:**
Prognostic Health Management aims to predict the Remaining Useful Life (RUL) of degrading components/systems utilizing monitoring data. These RUL predictions form the basis for optimizing maintenance planning in a Predictive Maintenance (PdM) paradigm. We here propose a metric for assessing data-driven prognostic algorithms based on their impact on downstream PdM decisions. The metric is defined in association with a decision setting and a corresponding PdM policy. We consider two typical PdM decision settings, namely component ordering and/or replacement planning, for which we investigate and improve PdM policies that are commonly utilized in the literature. All policies are evaluated via the data-based estimation of the long-run expected maintenance cost per unit time, using monitored run-to-failure experiments. The policy evaluation enables the estimation of the proposed metric. We employ the metric as an objective function for optimizing heuristic PdM policies and algorithms' hyperparameters. The effect of different PdM policies on the metric is initially investigated through a theoretical numerical example. Subsequently, we employ four data-driven prognostic algorithms on a simulated turbofan engine degradation problem, and investigate the joint effect of prognostic algorithm and PdM policy on the metric, resulting in a decision-oriented performance assessment of these algorithms.
       


### [Explainable Predictive Maintenance](https://arxiv.org/abs/2306.05120)

**Authors:**
Sepideh Pashami, Slawomir Nowaczyk, Yuantao Fan, Jakub Jakubowski, Nuno Paiva, Narjes Davari, Szymon Bobek, Samaneh Jamshidi, Hamid Sarmadi, Abdallah Alabdallah, Rita P. Ribeiro, Bruno Veloso, Moamar Sayed-Mouchaweh, Lala Rajaoarisoa, Grzegorz J. Nalepa, João Gama

**Abstract:**
Explainable Artificial Intelligence (XAI) fills the role of a critical interface fostering interactions between sophisticated intelligent systems and diverse individuals, including data scientists, domain experts, end-users, and more. It aids in deciphering the intricate internal mechanisms of ``black box'' Machine Learning (ML), rendering the reasons behind their decisions more understandable. However, current research in XAI primarily focuses on two aspects; ways to facilitate user trust, or to debug and refine the ML model. The majority of it falls short of recognising the diverse types of explanations needed in broader contexts, as different users and varied application areas necessitate solutions tailored to their specific needs.
  One such domain is Predictive Maintenance (PdM), an exploding area of research under the Industry 4.0 \& 5.0 umbrella. This position paper highlights the gap between existing XAI methodologies and the specific requirements for explanations within industrial applications, particularly the Predictive Maintenance field. Despite explainability's crucial role, this subject remains a relatively under-explored area, making this paper a pioneering attempt to bring relevant challenges to the research community's attention. We provide an overview of predictive maintenance tasks and accentuate the need and varying purposes for corresponding explanations. We then list and describe XAI techniques commonly employed in the literature, discussing their suitability for PdM tasks. Finally, to make the ideas and claims more concrete, we demonstrate XAI applied in four specific industrial use cases: commercial vehicles, metro trains, steel plants, and wind farms, spotlighting areas requiring further research.
       


### [Remaining Useful Life Modelling with an Escalator Health Condition Analytic System](https://arxiv.org/abs/2306.05436)

**Authors:**
Inez M. Zwetsloot, Yu Lin, Jiaqi Qiu, Lishuai Li, William Ka Fai Lee, Edmond Yin San Yeung, Colman Yiu Wah Yeung, Chris Chun Long Wong

**Abstract:**
The refurbishment of an escalator is usually linked with its design life as recommended by the manufacturer. However, the actual useful life of an escalator should be determined by its operating condition which is affected by the runtime, workload, maintenance quality, vibration, etc., rather than age only. The objective of this project is to develop a comprehensive health condition analytic system for escalators to support refurbishment decisions. The analytic system consists of four parts: 1) online data gathering and processing; 2) a dashboard for condition monitoring; 3) a health index model; and 4) remaining useful life prediction. The results can be used for a) predicting the remaining useful life of the escalators, in order to support asset replacement planning and b) monitoring the real-time condition of escalators; including alerts when vibration exceeds the threshold and signal diagnosis, giving an indication of possible root cause (components) of the alert signal.
       


### [Well-Calibrated Probabilistic Predictive Maintenance using Venn-Abers](https://arxiv.org/abs/2306.06642)

**Authors:**
Ulf Johansson, Tuwe Löfström, Cecilia Sönströd

**Abstract:**
When using machine learning for fault detection, a common problem is the fact that most data sets are very unbalanced, with the minority class (a fault) being the interesting one. In this paper, we investigate the usage of Venn-Abers predictors, looking specifically at the effect on the minority class predictions. A key property of Venn-Abers predictors is that they output well-calibrated probability intervals. In the experiments, we apply Venn-Abers calibration to decision trees, random forests and XGBoost models, showing how both overconfident and underconfident models are corrected. In addition, the benefit of using the valid probability intervals produced by Venn-Abers for decision support is demonstrated. When using techniques producing opaque underlying models, e.g., random forest and XGBoost, each prediction will consist of not only the label, but also a valid probability interval, where the width is an indication of the confidence in the estimate. Adding Venn-Abers on top of a decision tree allows inspection and analysis of the model, to understand both the underlying relationship, and finding out in which parts of feature space that the model is accurate and/or confident.
       


### [Current Trends in Digital Twin Development, Maintenance, and Operation: An Interview Study](https://arxiv.org/abs/2306.10085)

**Authors:**
Hossain Muhammad Muctadir, David A. Manrique Negrin, Raghavendran Gunasekaran, Loek Cleophas, Mark van den Brand, Boudewijn R. Haverkort

**Abstract:**
Digital twins (DT) are often defined as a pairing of a physical entity and a corresponding virtual entity (VE), mimicking certain aspects of the former depending on the use-case. In recent years, this concept has facilitated numerous use-cases ranging from design to validation and predictive maintenance of large and small high-tech systems. Various heterogeneous cross-domain models are essential for such systems and model-driven engineering plays a pivotal role in the design, development, and maintenance of these models. We believe models and model-driven engineering play a similarly crucial role in the context of a VE of a DT. Due to the rapidly growing popularity of DTs and their use in diverse domains and use-cases, the methodologies, tools, and practices for designing, developing, and maintaining the corresponding VEs differ vastly. To better understand these differences and similarities, we performed a semi-structured interview research with 19 professionals from industry and academia who are closely associated with different lifecycle stages of digital twins. In this paper, we present our analysis and findings from this study, which is based on seven research questions. In general, we identified an overall lack of uniformity in terms of the understanding of digital twins and used tools, techniques, and methodologies for the development and maintenance of the corresponding VEs. Furthermore, considering that digital twins are software intensive systems, we recognize a significant growth potential for adopting more software engineering practices, processes, and expertise in various stages of a digital twin's lifecycle.
       


### [Artificial Intelligence for Technical Debt Management in Software Development](https://arxiv.org/abs/2306.10194)

**Authors:**
Srinivas Babu Pandi, Samia A. Binta, Savita Kaushal

**Abstract:**
Technical debt is a well-known challenge in software development, and its negative impact on software quality, maintainability, and performance is widely recognized. In recent years, artificial intelligence (AI) has proven to be a promising approach to assist in managing technical debt. This paper presents a comprehensive literature review of existing research on the use of AI powered tools for technical debt avoidance in software development. In this literature review we analyzed 15 related research papers which covers various AI-powered techniques, such as code analysis and review, automated testing, code refactoring, predictive maintenance, code generation, and code documentation, and explores their effectiveness in addressing technical debt. The review also discusses the benefits and challenges of using AI for technical debt management, provides insights into the current state of research, and highlights gaps and opportunities for future research. The findings of this review suggest that AI has the potential to significantly improve technical debt management in software development, and that existing research provides valuable insights into how AI can be leveraged to address technical debt effectively and efficiently. However, the review also highlights several challenges and limitations of current approaches, such as the need for high-quality data and ethical considerations and underscores the importance of further research to address these issues. The paper provides a comprehensive overview of the current state of research on AI for technical debt avoidance and offers practical guidance for software development teams seeking to leverage AI in their development processes to mitigate technical debt effectively
       


### [Application of Deep Learning for Predictive Maintenance of Oilfield Equipment](https://arxiv.org/abs/2306.11040)

**Author:**
Abdeldjalil Latrach

**Abstract:**
This thesis explored applications of the new emerging techniques of artificial intelligence and deep learning (neural networks in particular) for predictive maintenance, diagnostics and prognostics. Many neural architectures such as fully-connected, convolutional and recurrent neural networks were developed and tested on public datasets such as NASA C-MAPSS, Case Western Reserve University Bearings and FEMTO Bearings datasets to diagnose equipment health state and/or predict the remaining useful life (RUL) before breakdown. Many data processing and feature extraction procedures were used in combination with deep learning techniques such as dimensionality reduction (Principal Component Analysis) and signal processing (Fourier and Wavelet analyses) in order to create more meaningful and robust features to use as an input for neural networks architectures. This thesis also explored the potential use of these techniques in predictive maintenance within oil rigs for monitoring oilfield critical equipment in order to reduce unpredicted downtime and maintenance costs.
       


### [Automated Machine Learning for Remaining Useful Life Predictions](https://arxiv.org/abs/2306.12215)

**Authors:**
Marc-André Zöller, Fabian Mauthe, Peter Zeiler, Marius Lindauer, Marco F. Huber

**Abstract:**
Being able to predict the remaining useful life (RUL) of an engineering system is an important task in prognostics and health management. Recently, data-driven approaches to RUL predictions are becoming prevalent over model-based approaches since no underlying physical knowledge of the engineering system is required. Yet, this just replaces required expertise of the underlying physics with machine learning (ML) expertise, which is often also not available. Automated machine learning (AutoML) promises to build end-to-end ML pipelines automatically enabling domain experts without ML expertise to create their own models. This paper introduces AutoRUL, an AutoML-driven end-to-end approach for automatic RUL predictions. AutoRUL combines fine-tuned standard regression methods to an ensemble with high predictive power. By evaluating the proposed method on eight real-world and synthetic datasets against state-of-the-art hand-crafted models, we show that AutoML provides a viable alternative to hand-crafted data-driven RUL predictions. Consequently, creating RUL predictions can be made more accessible for domain experts using AutoML by eliminating ML expertise from data-driven model construction.
       


### [Performance Analysis of Empirical Open-Circuit Voltage Modeling in Lithium Ion Batteries, Part-3: Experimental Results](https://arxiv.org/abs/2306.16575)

**Authors:**
Prarthana Pillai, James Nguyen, Balakumar Balasingam

**Abstract:**
This paper is the third part of a series of papers about empirical approaches to open circuit voltage (OCV) modeling of lithium-ion batteries. The first part of the series proposed models to quantify various sources of uncertainties in the OCV models; and, the second part of the series presented systematic data collection approaches to compute the uncertainties in the OCV-SOC models. This paper uses data collected from 28 OCV characterization experiments, performed according to the data collection plan presented, to compute and analyze the following three different OCV uncertainty metrics: cell-to-cell variations, cycle-rate error, and curve fitting error. From the computed metrics, it was observed that a lower C-Rate showed smaller errors in the OCV-SOC model and vice versa. The results reported in this paper establish a relationship between the C-Rate and the uncertainty of the OCV-SOC model. This research can be thus useful to battery researchers for quantifying the tradeoff between the time taken to complete the OCV characterization test and the corresponding uncertainty in the OCV-SOC modeling. Further, quantified uncertainty model parameters can be used to accurately characterize the uncertainty in various battery management functionalities, such as state of charge and state of health estimation.
       


## July
### [An End-To-End Analysis of Deep Learning-Based Remaining Useful Life Algorithms for Satefy-Critical 5G-Enabled IIoT Networks](https://arxiv.org/abs/2307.04632)

**Authors:**
Lorenzo Mario Amorosa, Nicolò Longhi, Giampaolo Cuozzo, Weronika Maria Bachan, Valerio Lieti, Enrico Buracchini, Roberto Verdone

**Abstract:**
Remaining Useful Life (RUL) prediction is a critical task that aims to estimate the amount of time until a system fails, where the latter is formed by three main components, that is, the application, communication network, and RUL logic. In this paper, we provide an end-to-end analysis of an entire RUL-based chain. Specifically, we consider a factory floor where Automated Guided Vehicles (AGVs) transport dangerous liquids whose fall may cause injuries to workers. Regarding the communication infrastructure, the AGVs are equipped with 5G User Equipments (UEs) that collect real-time data of their movements and send them to an application server. The RUL logic consists of a Deep Learning (DL)-based pipeline that assesses if there will be liquid falls by analyzing the collected data, and, eventually, sending commands to the AGVs to avoid such a danger. According to this scenario, we performed End-to-End 5G NR-compliant network simulations to study the Round-Trip Time (RTT) as a function of the overall system bandwidth, subcarrier spacing, and modulation order. Then, via real-world experiments, we collect data to train, test and compare 7 DL models and 1 baseline threshold-based algorithm in terms of cost and average advance. Finally, we assess whether or not the RTT provided by four different 5G NR network architectures is compatible with the average advance provided by the best-performing one-Dimensional Convolutional Neural Network (1D-CNN). Numerical results show under which conditions the DL-based approach for RUL estimation matches with the RTT performance provided by different 5G NR network architectures.
       


### [A Mapping Study of Machine Learning Methods for Remaining Useful Life Estimation of Lead-Acid Batteries](https://arxiv.org/abs/2307.05163)

**Authors:**
Sérgio F Chevtchenko, Elisson da Silva Rocha, Bruna Cruz, Ermeson Carneiro de Andrade, Danilo Ricardo Barbosa de Araújo

**Abstract:**
Energy storage solutions play an increasingly important role in modern infrastructure and lead-acid batteries are among the most commonly used in the rechargeable category. Due to normal degradation over time, correctly determining the battery's State of Health (SoH) and Remaining Useful Life (RUL) contributes to enhancing predictive maintenance, reliability, and longevity of battery systems. Besides improving the cost savings, correct estimation of the SoH can lead to reduced pollution though reuse of retired batteries. This paper presents a mapping study of the state-of-the-art in machine learning methods for estimating the SoH and RUL of lead-acid batteries. These two indicators are critical in the battery management systems of electric vehicles, renewable energy systems, and other applications that rely heavily on this battery technology. In this study, we analyzed the types of machine learning algorithms employed for estimating SoH and RUL, and evaluated their performance in terms of accuracy and inference time. Additionally, this mapping identifies and analyzes the most commonly used combinations of sensors in specific applications, such as vehicular batteries. The mapping concludes by highlighting potential gaps and opportunities for future research, which lays the foundation for further advancements in the field.
       


### [A Comprehensive Survey of Deep Transfer Learning for Anomaly Detection in Industrial Time Series: Methods, Applications, and Directions](https://arxiv.org/abs/2307.05638)

**Authors:**
Peng Yan, Ahmed Abdulkadir, Paul-Philipp Luley, Matthias Rosenthal, Gerrit A. Schatte, Benjamin F. Grewe, Thilo Stadelmann

**Abstract:**
Automating the monitoring of industrial processes has the potential to enhance efficiency and optimize quality by promptly detecting abnormal events and thus facilitating timely interventions. Deep learning, with its capacity to discern non-trivial patterns within large datasets, plays a pivotal role in this process. Standard deep learning methods are suitable to solve a specific task given a specific type of data. During training, deep learning demands large volumes of labeled data. However, due to the dynamic nature of the industrial processes and environment, it is impractical to acquire large-scale labeled data for standard deep learning training for every slightly different case anew. Deep transfer learning offers a solution to this problem. By leveraging knowledge from related tasks and accounting for variations in data distributions, the transfer learning framework solves new tasks with little or even no additional labeled data. The approach bypasses the need to retrain a model from scratch for every new setup and dramatically reduces the labeled data requirement. This survey first provides an in-depth review of deep transfer learning, examining the problem settings of transfer learning and classifying the prevailing deep transfer learning methods. Moreover, we delve into applications of deep transfer learning in the context of a broad spectrum of time series anomaly detection tasks prevalent in primary industrial domains, e.g., manufacturing process monitoring, predictive maintenance, energy management, and infrastructure facility monitoring. We discuss the challenges and limitations of deep transfer learning in industrial contexts and conclude the survey with practical directions and actionable suggestions to address the need to leverage diverse time series data for anomaly detection in an increasingly dynamic production environment.
       


### [Predicting Battery Lifetime Under Varying Usage Conditions from Early Aging Data](https://arxiv.org/abs/2307.08382)

**Authors:**
Tingkai Li, Zihao Zhou, Adam Thelen, David Howey, Chao Hu

**Abstract:**
Accurate battery lifetime prediction is important for preventative maintenance, warranties, and improved cell design and manufacturing. However, manufacturing variability and usage-dependent degradation make life prediction challenging. Here, we investigate new features derived from capacity-voltage data in early life to predict the lifetime of cells cycled under widely varying charge rates, discharge rates, and depths of discharge. Features were extracted from regularly scheduled reference performance tests (i.e., low rate full cycles) during cycling. The early-life features capture a cell's state of health and the rate of change of component-level degradation modes, some of which correlate strongly with cell lifetime. Using a newly generated dataset from 225 nickel-manganese-cobalt/graphite Li-ion cells aged under a wide range of conditions, we demonstrate a lifetime prediction of in-distribution cells with 15.1% mean absolute percentage error using no more than the first 15% of data, for most cells. Further testing using a hierarchical Bayesian regression model shows improved performance on extrapolation, achieving 21.8% mean absolute percentage error for out-of-distribution cells. Our approach highlights the importance of using domain knowledge of lithium-ion battery degradation modes to inform feature engineering. Further, we provide the community with a new publicly available battery aging dataset with cells cycled beyond 80% of their rated capacity.
       


### [Demonstration of a Response Time Based Remaining Useful Life (RUL) Prediction for Software Systems](https://arxiv.org/abs/2307.12237)

**Authors:**
Ray Islam, Peter Sandborn

**Abstract:**
Prognostic and Health Management (PHM) has been widely applied to hardware systems in the electronics and non-electronics domains but has not been explored for software. While software does not decay over time, it can degrade over release cycles. Software health management is confined to diagnostic assessments that identify problems, whereas prognostic assessment potentially indicates when in the future a problem will become detrimental. Relevant research areas such as software defect prediction, software reliability prediction, predictive maintenance of software, software degradation, and software performance prediction, exist, but all of these represent diagnostic models built upon historical data, none of which can predict an RUL for software. This paper addresses the application of PHM concepts to software systems for fault predictions and RUL estimation. Specifically, this paper addresses how PHM can be used to make decisions for software systems such as version update and upgrade, module changes, system reengineering, rejuvenation, maintenance scheduling, budgeting, and total abandonment. This paper presents a method to prognostically and continuously predict the RUL of a software system based on usage parameters (e.g., the numbers and categories of releases) and performance parameters (e.g., response time). The model developed has been validated by comparing actual data, with the results that were generated by predictive models. Statistical validation (regression validation, and k-fold cross validation) has also been carried out. A case study, based on publicly available data for the Bugzilla application is presented. This case study demonstrates that PHM concepts can be applied to software systems and RUL can be calculated to make system management decisions.
       


### [Robustness Verification of Deep Neural Networks using Star-Based Reachability Analysis with Variable-Length Time Series Input](https://arxiv.org/abs/2307.13907)

**Authors:**
Neelanjana Pal, Diego Manzanas Lopez, Taylor T Johnson

**Abstract:**
Data-driven, neural network (NN) based anomaly detection and predictive maintenance are emerging research areas. NN-based analytics of time-series data offer valuable insights into past behaviors and estimates of critical parameters like remaining useful life (RUL) of equipment and state-of-charge (SOC) of batteries. However, input time series data can be exposed to intentional or unintentional noise when passing through sensors, necessitating robust validation and verification of these NNs. This paper presents a case study of the robustness verification approach for time series regression NNs (TSRegNN) using set-based formal methods. It focuses on utilizing variable-length input data to streamline input manipulation and enhance network architecture generalizability. The method is applied to two data sets in the Prognostics and Health Management (PHM) application areas: (1) SOC estimation of a Lithium-ion battery and (2) RUL estimation of a turbine engine. The NNs' robustness is checked using star-based reachability analysis, and several performance measures evaluate the effect of bounded perturbations in the input on network outputs, i.e., future outcomes. Overall, the paper offers a comprehensive case study for validating and verifying NN-based analytics of time-series data in real-world applications, emphasizing the importance of robustness testing for accurate and reliable predictions, especially considering the impact of noise on future outcomes.
       


### [Predictive Maintenance of Armoured Vehicles using Machine Learning Approaches](https://arxiv.org/abs/2307.14453)

**Authors:**
Prajit Sengupta, Anant Mehta, Prashant Singh Rana

**Abstract:**
Armoured vehicles are specialized and complex pieces of machinery designed to operate in high-stress environments, often in combat or tactical situations. This study proposes a predictive maintenance-based ensemble system that aids in predicting potential maintenance needs based on sensor data collected from these vehicles. The proposed model's architecture involves various models such as Light Gradient Boosting, Random Forest, Decision Tree, Extra Tree Classifier and Gradient Boosting to predict the maintenance requirements of the vehicles accurately. In addition, K-fold cross validation, along with TOPSIS analysis, is employed to evaluate the proposed ensemble model's stability. The results indicate that the proposed system achieves an accuracy of 98.93%, precision of 99.80% and recall of 99.03%. The algorithm can effectively predict maintenance needs, thereby reducing vehicle downtime and improving operational efficiency. Through comparisons between various algorithms and the suggested ensemble, this study highlights the potential of machine learning-based predictive maintenance solutions.
       


## August
### [Deep Reinforcement Learning-Based Battery Conditioning Hierarchical V2G Coordination for Multi-Stakeholder Benefits](https://arxiv.org/abs/2308.00218)

**Authors:**
Yubao Zhang, Xin Chen, Yi Gu, Zhicheng Li, Wu Kai

**Abstract:**
With the growing prevalence of electric vehicles (EVs) and advancements in EV electronics, vehicle-to-grid (V2G) techniques and large-scale scheduling strategies have emerged to promote renewable energy utilization and power grid stability. This study proposes a multi-stakeholder hierarchical V2G coordination based on deep reinforcement learning (DRL) and the Proof of Stake algorithm. Furthermore, the multi-stakeholders include the power grid, EV aggregators (EVAs), and users, and the proposed strategy can achieve multi-stakeholder benefits. On the grid side, load fluctuations and renewable energy consumption are considered, while on the EVA side, energy constraints and charging costs are considered. The three critical battery conditioning parameters of battery SOX are considered on the user side, including state of charge, state of power, and state of health. Compared with four typical baselines, the multi-stakeholder hierarchical coordination strategy can enhance renewable energy consumption, mitigate load fluctuations, meet the energy demands of EVA, and reduce charging costs and battery degradation under realistic operating conditions.
       


### [A digital twin framework for civil engineering structures](https://arxiv.org/abs/2308.01445)

**Authors:**
Matteo Torzoni, Marco Tezzele, Stefano Mariani, Andrea Manzoni, Karen E. Willcox

**Abstract:**
The digital twin concept represents an appealing opportunity to advance condition-based and predictive maintenance paradigms for civil engineering systems, thus allowing reduced lifecycle costs, increased system safety, and increased system availability. This work proposes a predictive digital twin approach to the health monitoring, maintenance, and management planning of civil engineering structures. The asset-twin coupled dynamical system is encoded employing a probabilistic graphical model, which allows all relevant sources of uncertainty to be taken into account. In particular, the time-repeating observations-to-decisions flow is modeled using a dynamic Bayesian network. Real-time structural health diagnostics are provided by assimilating sensed data with deep learning models. The digital twin state is continually updated in a sequential Bayesian inference fashion. This is then exploited to inform the optimal planning of maintenance and management actions within a dynamic decision-making framework. A preliminary offline phase involves the population of training datasets through a reduced-order numerical model and the computation of a health-dependent control policy. The strategy is assessed on two synthetic case studies, involving a cantilever beam and a railway bridge, demonstrating the dynamic decision-making capabilities of health-aware digital twins.
       


### [Deep Koopman Operator-based degradation modelling](https://arxiv.org/abs/2308.01690)

**Authors:**
Sergei Garmaev, Olga Fink

**Abstract:**
With the current trend of increasing complexity of industrial systems, the construction and monitoring of health indicators becomes even more challenging. Given that health indicators are commonly employed to predict the end of life, a crucial criterion for reliable health indicators is their capability to discern a degradation trend. However, trending can pose challenges due to the variability of operating conditions. An optimal transformation of health indicators would therefore be one that converts degradation dynamics into a coordinate system where degradation trends exhibit linearity. Koopman theory framework is well-suited to address these challenges. In this work, we demonstrate the successful extension of the previously proposed Deep Koopman Operator approach to learn the dynamics of industrial systems by transforming them into linearized coordinate systems, resulting in a latent representation that provides sufficient information for estimating the system's remaining useful life. Additionally, we propose a novel Koopman-Inspired Degradation Model for degradation modelling of dynamical systems with control. The proposed approach effectively disentangles the impact of degradation and imposed control on the latent dynamics. The algorithm consistently outperforms in predicting the remaining useful life of CNC milling machine cutters and Li-ion batteries, whether operated under constant and varying current loads. Furthermore, we highlight the utility of learned Koopman-inspired degradation operators analyzing the influence of imposed control on the system's health state.
       


### [Applications and Societal Implications of Artificial Intelligence in Manufacturing: A Systematic Review](https://arxiv.org/abs/2308.02025)

**Authors:**
John P. Nelson, Justin B. Biddle, Philip Shapira

**Abstract:**
This paper undertakes a systematic review of relevant extant literature to consider the potential societal implications of the growth of AI in manufacturing. We analyze the extensive range of AI applications in this domain, such as interfirm logistics coordination, firm procurement management, predictive maintenance, and shop-floor monitoring and control of processes, machinery, and workers. Additionally, we explore the uncertain societal implications of industrial AI, including its impact on the workforce, job upskilling and deskilling, cybersecurity vulnerability, and environmental consequences. After building a typology of AI applications in manufacturing, we highlight the diverse possibilities for AI's implementation at different scales and application types. We discuss the importance of considering AI's implications both for individual firms and for society at large, encompassing economic prosperity, equity, environmental health, and community safety and security. The study finds that there is a predominantly optimistic outlook in prior literature regarding AI's impact on firms, but that there is substantial debate and contention about adverse effects and the nature of AI's societal implications. The paper draws analogies to historical cases and other examples to provide a contextual perspective on potential societal effects of industrial AI. Ultimately, beneficial integration of AI in manufacturing will depend on the choices and priorities of various stakeholders, including firms and their managers and owners, technology developers, civil society organizations, and governments. A broad and balanced awareness of opportunities and risks among stakeholders is vital not only for successful and safe technical implementation but also to construct a socially beneficial and sustainable future for manufacturing in the age of AI.
       


### [Causal Disentanglement Hidden Markov Model for Fault Diagnosis](https://arxiv.org/abs/2308.03027)

**Authors:**
Rihao Chang, Yongtao Ma, Weizhi Nie, Jie Nie, An-an Liu

**Abstract:**
In modern industries, fault diagnosis has been widely applied with the goal of realizing predictive maintenance. The key issue for the fault diagnosis system is to extract representative characteristics of the fault signal and then accurately predict the fault type. In this paper, we propose a Causal Disentanglement Hidden Markov model (CDHM) to learn the causality in the bearing fault mechanism and thus, capture their characteristics to achieve a more robust representation. Specifically, we make full use of the time-series data and progressively disentangle the vibration signal into fault-relevant and fault-irrelevant factors. The ELBO is reformulated to optimize the learning of the causal disentanglement Markov model. Moreover, to expand the scope of the application, we adopt unsupervised domain adaptation to transfer the learned disentangled representations to other working environments. Experiments were conducted on the CWRU dataset and IMS dataset. Relevant results validate the superiority of the proposed method.
       


### [Two-stage Early Prediction Framework of Remaining Useful Life for Lithium-ion Batteries](https://arxiv.org/abs/2308.03664)

**Authors:**
Dhruv Mittal, Hymalai Bello, Bo Zhou, Mayank Shekhar Jha, Sungho Suh, Paul Lukowicz

**Abstract:**
Early prediction of remaining useful life (RUL) is crucial for effective battery management across various industries, ranging from household appliances to large-scale applications. Accurate RUL prediction improves the reliability and maintainability of battery technology. However, existing methods have limitations, including assumptions of data from the same sensors or distribution, foreknowledge of the end of life (EOL), and neglect to determine the first prediction cycle (FPC) to identify the start of the unhealthy stage. This paper proposes a novel method for RUL prediction of Lithium-ion batteries. The proposed framework comprises two stages: determining the FPC using a neural network-based model to divide the degradation data into distinct health states and predicting the degradation pattern after the FPC to estimate the remaining useful life as a percentage. Experimental results demonstrate that the proposed method outperforms conventional approaches in terms of RUL prediction. Furthermore, the proposed method shows promise for real-world scenarios, providing improved accuracy and applicability for battery management.
       


### [Deep convolutional neural networks for cyclic sensor data](https://arxiv.org/abs/2308.06987)

**Authors:**
Payman Goodarzi, Yannick Robin, Andreas Schütze, Tizian Schneider

**Abstract:**
Predictive maintenance plays a critical role in ensuring the uninterrupted operation of industrial systems and mitigating the potential risks associated with system failures. This study focuses on sensor-based condition monitoring and explores the application of deep learning techniques using a hydraulic system testbed dataset. Our investigation involves comparing the performance of three models: a baseline model employing conventional methods, a single CNN model with early sensor fusion, and a two-lane CNN model (2L-CNN) with late sensor fusion. The baseline model achieves an impressive test error rate of 1% by employing late sensor fusion, where feature extraction is performed individually for each sensor. However, the CNN model encounters challenges due to the diverse sensor characteristics, resulting in an error rate of 20.5%. To further investigate this issue, we conduct separate training for each sensor and observe variations in accuracy. Additionally, we evaluate the performance of the 2L-CNN model, which demonstrates significant improvement by reducing the error rate by 33% when considering the combination of the least and most optimal sensors. This study underscores the importance of effectively addressing the complexities posed by multi-sensor systems in sensor-based condition monitoring.
       


### [A Transformer-based Framework For Multi-variate Time Series: A Remaining Useful Life Prediction Use Case](https://arxiv.org/abs/2308.09884)

**Authors:**
Oluwaseyi Ogunfowora, Homayoun Najjaran

**Abstract:**
In recent times, Large Language Models (LLMs) have captured a global spotlight and revolutionized the field of Natural Language Processing. One of the factors attributed to the effectiveness of LLMs is the model architecture used for training, transformers. Transformer models excel at capturing contextual features in sequential data since time series data are sequential, transformer models can be leveraged for more efficient time series data prediction. The field of prognostics is vital to system health management and proper maintenance planning. A reliable estimation of the remaining useful life (RUL) of machines holds the potential for substantial cost savings. This includes avoiding abrupt machine failures, maximizing equipment usage, and serving as a decision support system (DSS). This work proposed an encoder-transformer architecture-based framework for multivariate time series prediction for a prognostics use case. We validated the effectiveness of the proposed framework on all four sets of the C-MAPPS benchmark dataset for the remaining useful life prediction task. To effectively transfer the knowledge and application of transformers from the natural language domain to time series, three model-specific experiments were conducted. Also, to enable the model awareness of the initial stages of the machine life and its degradation path, a novel expanding window method was proposed for the first time in this work, it was compared with the sliding window method, and it led to a large improvement in the performance of the encoder transformer model. Finally, the performance of the proposed encoder-transformer model was evaluated on the test dataset and compared with the results from 13 other state-of-the-art (SOTA) models in the literature and it outperformed them all with an average performance increase of 137.65% over the next best model across all the datasets.
       


### [On-Premise AIOps Infrastructure for a Software Editor SME: An Experience Report](https://arxiv.org/abs/2308.11225)

**Authors:**
Anes Bendimerad, Youcef Remil, Romain Mathonat, Mehdi Kaytoue

**Abstract:**
Information Technology has become a critical component in various industries, leading to an increased focus on software maintenance and monitoring. With the complexities of modern software systems, traditional maintenance approaches have become insufficient. The concept of AIOps has emerged to enhance predictive maintenance using Big Data and Machine Learning capabilities. However, exploiting AIOps requires addressing several challenges related to the complexity of data and incident management. Commercial solutions exist, but they may not be suitable for certain companies due to high costs, data governance issues, and limitations in covering private software. This paper investigates the feasibility of implementing on-premise AIOps solutions by leveraging open-source tools. We introduce a comprehensive AIOps infrastructure that we have successfully deployed in our company, and we provide the rationale behind different choices that we made to build its various components. Particularly, we provide insights into our approach and criteria for selecting a data management system and we explain its integration. Our experience can be beneficial for companies seeking to internally manage their software maintenance processes with a modern AIOps approach.
       


### [Comprehensive performance comparison among different types of features in data-driven battery state of health estimation](https://arxiv.org/abs/2308.13993)

**Authors:**
Xinhong Feng, Yongzhi Zhang, Rui Xiong, Chun Wang

**Abstract:**
Battery state of health (SOH), which informs the maximal available capacity of the battery, is a key indicator of battery aging failure. Accurately estimating battery SOH is a vital function of the battery management system that remains to be addressed. In this study, a physics-informed Gaussian process regression (GPR) model is developed for battery SOH estimation, with the performance being systematically compared with that of different types of features and machine learning (ML) methods. The method performance is validated based on 58826 cycling data units of 118 cells. Experimental results show that the physics-driven ML generally estimates more accurate SOH than other non-physical features under different scenarios. The physical features-based GPR predicts battery SOH with the errors being less than 1.1% based on 10 to 20 mins' relaxation data. And the high robustness and generalization capability of the methodology is also validated against different ratios of training and test data under unseen conditions. Results also highlight the more effective capability of knowledge transfer between different types of batteries with the physical features and GPR. This study demonstrates the excellence of physical features in indicating the state evolution of complex systems, and the improved indication performance of these features by combining a suitable ML method.
       


### [Compartment model with retarded transition rates](https://arxiv.org/abs/2308.14495)

**Authors:**
Teo Granger, Thomas Michelitsch, Bernard Collet, Michael Bestehorn, Alejandro Riascos

**Abstract:**
Our study is devoted to a four-compartment epidemic model of a constant population of independent random walkers. Each walker is in one of four compartments (S-susceptible, C-infected but not infectious (period of incubation), I-infected and infectious, R-recovered and immune) characterizing the states of health. The walkers navigate independently on a periodic 2D lattice. Infections occur by collisions of susceptible and infectious walkers. Once infected, a walker undergoes the delayed cyclic transition pathway S $\to$ C $\to$ I $\to$ R $\to$ S. The random delay times between the transitions (sojourn times in the compartments) are drawn from independent probability density functions (PDFs). We analyze the existence of the endemic equilibrium and stability of the globally healthy state and derive a condition for the spread of the epidemics which we connect with the basic reproduction number $R_0>1$. We give quantitative numerical evidence that a simple approach based on random walkers offers an appropriate microscopic picture of the dynamics for this class of epidemics.
       


## September
### [TFBEST: Dual-Aspect Transformer with Learnable Positional Encoding for Failure Prediction](https://arxiv.org/abs/2309.02641)

**Authors:**
Rohan Mohapatra, Saptarshi Sengupta

**Abstract:**
Hard Disk Drive (HDD) failures in datacenters are costly - from catastrophic data loss to a question of goodwill, stakeholders want to avoid it like the plague. An important tool in proactively monitoring against HDD failure is timely estimation of the Remaining Useful Life (RUL). To this end, the Self-Monitoring, Analysis and Reporting Technology employed within HDDs (S.M.A.R.T.) provide critical logs for long-term maintenance of the security and dependability of these essential data storage devices. Data-driven predictive models in the past have used these S.M.A.R.T. logs and CNN/RNN based architectures heavily. However, they have suffered significantly in providing a confidence interval around the predicted RUL values as well as in processing very long sequences of logs. In addition, some of these approaches, such as those based on LSTMs, are inherently slow to train and have tedious feature engineering overheads. To overcome these challenges, in this work we propose a novel transformer architecture - a Temporal-fusion Bi-encoder Self-attention Transformer (TFBEST) for predicting failures in hard-drives. It is an encoder-decoder based deep learning technique that enhances the context gained from understanding health statistics sequences and predicts a sequence of the number of days remaining before a disk potentially fails. In this paper, we also provide a novel confidence margin statistic that can help manufacturers replace a hard-drive within a time frame. Experiments on Seagate HDD data show that our method significantly outperforms the state-of-the-art RUL prediction methods during testing over the exhaustive 10-year data from Backblaze (2013-present). Although validated on HDD failure prediction, the TFBEST architecture is well-suited for other prognostics applications and may be adapted for allied regression problems.
       


### [Robust-MBDL: A Robust Multi-branch Deep Learning Based Model for Remaining Useful Life Prediction and Operational Condition Identification of Rotating Machines](https://arxiv.org/abs/2309.06157)

**Authors:**
Khoa Tran, Hai-Canh Vu, Lam Pham, Nassim Boudaoud

**Abstract:**
In this paper, a Robust Multi-branch Deep learning-based system for remaining useful life (RUL) prediction and condition operations (CO) identification of rotating machines is proposed. In particular, the proposed system comprises main components: (1) an LSTM-Autoencoder to denoise the vibration data; (2) a feature extraction to generate time-domain, frequency-domain, and time-frequency based features from the denoised data; (3) a novel and robust multi-branch deep learning network architecture to exploit the multiple features. The performance of our proposed system was evaluated and compared to the state-of-the-art systems on two benchmark datasets of XJTU-SY and PRONOSTIA. The experimental results prove that our proposed system outperforms the state-of-the-art systems and presents potential for real-life applications on bearing machines.
       


### [Predicting Survival Time of Ball Bearings in the Presence of Censoring](https://arxiv.org/abs/2309.07188)

**Authors:**
Christian Marius Lillelund, Fernando Pannullo, Morten Opprud Jakobsen, Christian Fischer Pedersen

**Abstract:**
Ball bearings find widespread use in various manufacturing and mechanical domains, and methods based on machine learning have been widely adopted in the field to monitor wear and spot defects before they lead to failures. Few studies, however, have addressed the problem of censored data, in which failure is not observed. In this paper, we propose a novel approach to predict the time to failure in ball bearings using survival analysis. First, we analyze bearing data in the frequency domain and annotate when a bearing fails by comparing the Kullback-Leibler divergence and the standard deviation between its break-in frequency bins and its break-out frequency bins. Second, we train several survival models to estimate the time to failure based on the annotated data and covariates extracted from the time domain, such as skewness, kurtosis and entropy. The models give a probabilistic prediction of risk over time and allow us to compare the survival function between groups of bearings. We demonstrate our approach on the XJTU and PRONOSTIA datasets. On XJTU, the best result is a 0.70 concordance-index and 0.21 integrated Brier score. On PRONOSTIA, the best is a 0.76 concordance-index and 0.19 integrated Brier score. Our work motivates further work on incorporating censored data in models for predictive maintenance.
       


### [Prognosis of Multivariate Battery State of Performance and Health via Transformers](https://arxiv.org/abs/2309.10014)

**Authors:**
Noah H. Paulson, Joseph J. Kubal, Susan J. Babinec

**Abstract:**
Batteries are an essential component in a deeply decarbonized future. Understanding battery performance and "useful life" as a function of design and use is of paramount importance to accelerating adoption. Historically, battery state of health (SOH) was summarized by a single parameter, the fraction of a battery's capacity relative to its initial state. A more useful approach, however, is a comprehensive characterization of its state and complexities, using an interrelated set of descriptors including capacity, energy, ionic and electronic impedances, open circuit voltages, and microstructure metrics. Indeed, predicting across an extensive suite of properties as a function of battery use is a "holy grail" of battery science; it can provide unprecedented insights toward the design of better batteries with reduced experimental effort, and de-risking energy storage investments that are necessary to meet CO2 reduction targets. In this work, we present a first step in that direction via deep transformer networks for the prediction of 28 battery state of health descriptors using two cycling datasets representing six lithium-ion cathode chemistries (LFP, NMC111, NMC532, NMC622, HE5050, and 5Vspinel), multiple electrolyte/anode compositions, and different charge-discharge scenarios. The accuracy of these predictions versus battery life (with an unprecedented mean absolute error of 19 cycles in predicting end of life for an LFP fast-charging dataset) illustrates the promise of deep learning towards providing deeper understanding and control of battery health.
       


### [Exploring the Correlation Between Ultrasound Speed and the State of Health of LiFePO$_4$ Prismatic Cells](https://arxiv.org/abs/2309.12191)

**Authors:**
Shengyuan Zhang, Peng Zuo, Xuesong Yin, Zheng Fan

**Abstract:**
Electric vehicles (EVs) have become a popular mode of transportation, with their performance depending on the ageing of the Li-ion batteries used to power them. However, it can be challenging and time-consuming to determine the capacity retention of a battery in service. A rapid and reliable testing method for state of health (SoH) determination is desired. Ultrasonic testing techniques are promising due to their efficient, portable, and non-destructive features. In this study, we demonstrate that ultrasonic speed decreases with the degradation of the capacity of an LFP prismatic cell. We explain this correlation through numerical simulation, which describes wave propagation in porous media. We propose that the reduction of binder stiffness can be a primary cause of the change in ultrasonic speed during battery ageing. This work brings new insights into ultrasonic SoH estimation techniques.
       


### [Ensemble Neural Networks for Remaining Useful Life (RUL) Prediction](https://arxiv.org/abs/2309.12445)

**Authors:**
Ahbishek Srinivasan, Juan Carlos Andresen, Anders Holst

**Abstract:**
A core part of maintenance planning is a monitoring system that provides a good prognosis on health and degradation, often expressed as remaining useful life (RUL). Most of the current data-driven approaches for RUL prediction focus on single-point prediction. These point prediction approaches do not include the probabilistic nature of the failure. The few probabilistic approaches to date either include the aleatoric uncertainty (which originates from the system), or the epistemic uncertainty (which originates from the model parameters), or both simultaneously as a total uncertainty. Here, we propose ensemble neural networks for probabilistic RUL predictions which considers both uncertainties and decouples these two uncertainties. These decoupled uncertainties are vital in knowing and interpreting the confidence of the predictions. This method is tested on NASA's turbofan jet engine CMAPSS data-set. Our results show how these uncertainties can be modeled and how to disentangle the contribution of aleatoric and epistemic uncertainty. Additionally, our approach is evaluated on different metrics and compared against the current state-of-the-art methods.
       


### [Analyzing the Influence of Processor Speed and Clock Speed on Remaining Useful Life Estimation of Software Systems](https://arxiv.org/abs/2309.12617)

**Authors:**
M. Rubyet Islam, Peter Sandborn

**Abstract:**
Prognostics and Health Management (PHM) is a discipline focused on predicting the point at which systems or components will cease to perform as intended, typically measured as Remaining Useful Life (RUL). RUL serves as a vital decision-making tool for contingency planning, guiding the timing and nature of system maintenance. Historically, PHM has primarily been applied to hardware systems, with its application to software only recently explored. In a recent study we introduced a methodology and demonstrated how changes in software can impact the RUL of software. However, in practical software development, real-time performance is also influenced by various environmental attributes, including operating systems, clock speed, processor performance, RAM, machine core count and others. This research extends the analysis to assess how changes in environmental attributes, such as operating system and clock speed, affect RUL estimation in software. Findings are rigorously validated using real performance data from controlled test beds and compared with predictive model-generated data. Statistical validation, including regression analysis, supports the credibility of the results. The controlled test bed environment replicates and validates faults from real applications, ensuring a standardized assessment platform. This exploration yields actionable knowledge for software maintenance and optimization strategies, addressing a significant gap in the field of software health management.
       


### [An Interpretable Systematic Review of Machine Learning Models for Predictive Maintenance of Aircraft Engine](https://arxiv.org/abs/2309.13310)

**Authors:**
Abdullah Al Hasib, Ashikur Rahman, Mahpara Khabir, Md. Tanvir Rouf Shawon

**Abstract:**
This paper presents an interpretable review of various machine learning and deep learning models to predict the maintenance of aircraft engine to avoid any kind of disaster. One of the advantages of the strategy is that it can work with modest datasets. In this study, sensor data is utilized to predict aircraft engine failure within a predetermined number of cycles using LSTM, Bi-LSTM, RNN, Bi-RNN GRU, Random Forest, KNN, Naive Bayes, and Gradient Boosting. We explain how deep learning and machine learning can be used to generate predictions in predictive maintenance using a straightforward scenario with just one data source. We applied lime to the models to help us understand why machine learning models did not perform well than deep learning models. An extensive analysis of the model's behavior is presented for several test data to understand the black box scenario of the models. A lucrative accuracy of 97.8%, 97.14%, and 96.42% are achieved by GRU, Bi-LSTM, and LSTM respectively which denotes the capability of the models to predict maintenance at an early stage.
       


### [Driving behavior-guided battery health monitoring for electric vehicles using machine learning](https://arxiv.org/abs/2309.14125)

**Authors:**
Nanhua Jiang, Jiawei Zhang, Weiran Jiang, Yao Ren, Jing Lin, Edwin Khoo, Ziyou Song

**Abstract:**
An accurate estimation of the state of health (SOH) of batteries is critical to ensuring the safe and reliable operation of electric vehicles (EVs). Feature-based machine learning methods have exhibited enormous potential for rapidly and precisely monitoring battery health status. However, simultaneously using various health indicators (HIs) may weaken estimation performance due to feature redundancy. Furthermore, ignoring real-world driving behaviors can lead to inaccurate estimation results as some features are rarely accessible in practical scenarios. To address these issues, we proposed a feature-based machine learning pipeline for reliable battery health monitoring, enabled by evaluating the acquisition probability of features under real-world driving conditions. We first summarized and analyzed various individual HIs with mechanism-related interpretations, which provide insightful guidance on how these features relate to battery degradation modes. Moreover, all features were carefully evaluated and screened based on estimation accuracy and correlation analysis on three public battery degradation datasets. Finally, the scenario-based feature fusion and acquisition probability-based practicality evaluation method construct a useful tool for feature extraction with consideration of driving behaviors. This work highlights the importance of balancing the performance and practicality of HIs during the development of feature-based battery health monitoring algorithms.
       


### [TranDRL: A Transformer-Driven Deep Reinforcement Learning Enabled Prescriptive Maintenance Framework](https://arxiv.org/abs/2309.16935)

**Authors:**
Yang Zhao, Jiaxi Yang, Wenbo Wang, Helin Yang, Dusit Niyato

**Abstract:**
Industrial systems demand reliable predictive maintenance strategies to enhance operational efficiency and reduce downtime. This paper introduces an integrated framework that leverages the capabilities of the Transformer model-based neural networks and deep reinforcement learning (DRL) algorithms to optimize system maintenance actions. Our approach employs the Transformer model to effectively capture complex temporal patterns in sensor data, thereby accurately predicting the remaining useful life (RUL) of an equipment. Additionally, the DRL component of our framework provides cost-effective and timely maintenance recommendations. We validate the efficacy of our framework on the NASA C-MPASS dataset, where it demonstrates significant advancements in both RUL prediction accuracy and the optimization of maintenance actions, compared to the other prevalent machine learning-based methods. Our proposed approach provides an innovative data-driven framework for industry machine systems, accurately forecasting equipment lifespans and optimizing maintenance schedules, thereby reducing downtime and cutting costs.
       


### [Imagery Dataset for Condition Monitoring of Synthetic Fibre Ropes](https://arxiv.org/abs/2309.17058)

**Authors:**
Anju Rani, Daniel O. Arroyo, Petar Durdevic

**Abstract:**
Automatic visual inspection of synthetic fibre ropes (SFRs) is a challenging task in the field of offshore, wind turbine industries, etc. The presence of any defect in SFRs can compromise their structural integrity and pose significant safety risks. Due to the large size and weight of these ropes, it is often impractical to detach and inspect them frequently. Therefore, there is a critical need to develop efficient defect detection methods to assess their remaining useful life (RUL). To address this challenge, a comprehensive dataset has been generated, comprising a total of 6,942 raw images representing both normal and defective SFRs. The dataset encompasses a wide array of defect scenarios which may occur throughout their operational lifespan, including but not limited to placking defects, cut strands, chafings, compressions, core outs and normal. This dataset serves as a resource to support computer vision applications, including object detection, classification, and segmentation, aimed at detecting and analyzing defects in SFRs. The availability of this dataset will facilitate the development and evaluation of robust defect detection algorithms. The aim of generating this dataset is to assist in the development of automated defect detection systems that outperform traditional visual inspection methods, thereby paving the way for safer and more efficient utilization of SFRs across a wide range of applications.
       


### [Identifiability Study of Lithium-Ion Battery Capacity Fade Using Degradation Mode Sensitivity for a Minimally and Intuitively Parametrized Electrode-Specific Cell Open-Circuit Voltage Model](https://arxiv.org/abs/2309.17331)

**Authors:**
Jing Lin, Edwin Khoo

**Abstract:**
When two electrode open-circuit potentials form a full-cell OCV (open-circuit voltage) model, cell-level SOH (state of health) parameters related to LLI (loss of lithium inventory) and LAM (loss of active materials) naturally appear. Such models have been used to interpret experimental OCV measurements and infer these SOH parameters associated with capacity fade. In this work, we first re-parametrize a popular OCV model formulation by the N/P (negative-to-positive) ratio and Li/P (lithium-to-positive) ratio, which have more symmetric and intuitive physical meaning, and are also pristine-condition-agnostic and cutoff-voltage-independent. We then study the modal identifiability of capacity fade by mathematically deriving the gradients of electrode slippage and cell OCV with respect to these SOH parameters, where the electrode differential voltage fractions, which characterize each electrode's relative contribution to the OCV slope, play a key role in passing the influence of a fixed cutoff voltage to the parameter sensitivity. The sensitivity gradients of the total capacity also reveal four characteristic regimes regarding how much lithium inventory and active materials are limiting the apparent capacity. We show the usefulness of these sensitivity gradients with an application regarding degradation mode identifiability from OCV measurements at different SOC (state of charge) windows.
       


## October
### [De-SaTE: Denoising Self-attention Transformer Encoders for Li-ion Battery Health Prognostics](https://arxiv.org/abs/2310.00023)

**Authors:**
Gaurav Shinde, Rohan Mohapatra, Pooja Krishan, Saptarshi Sengupta

**Abstract:**
The usage of Lithium-ion (Li-ion) batteries has gained widespread popularity across various industries, from powering portable electronic devices to propelling electric vehicles and supporting energy storage systems. A central challenge in Li-ion battery reliability lies in accurately predicting their Remaining Useful Life (RUL), which is a critical measure for proactive maintenance and predictive analytics. This study presents a novel approach that harnesses the power of multiple denoising modules, each trained to address specific types of noise commonly encountered in battery data. Specifically, a denoising auto-encoder and a wavelet denoiser are used to generate encoded/decomposed representations, which are subsequently processed through dedicated self-attention transformer encoders. After extensive experimentation on NASA and CALCE data, a broad spectrum of health indicator values are estimated under a set of diverse noise patterns. The reported error metrics on these data are on par with or better than the state-of-the-art reported in recent literature.
       


### [Managing the Impact of Sensor's Thermal Noise in Machine Learning for Nuclear Applications](https://arxiv.org/abs/2310.01014)

**Author:**
Issam Hammad

**Abstract:**
Sensors such as accelerometers, magnetometers, and gyroscopes are frequently utilized to perform measurements in nuclear power plants. For example, accelerometers are used for vibration monitoring of critical systems. With the recent rise of machine learning, data captured from such sensors can be used to build machine learning models for predictive maintenance and automation. However, these sensors are known to have thermal noise that can affect the sensor's accuracy. Thermal noise differs between sensors in terms of signal-to-noise ratio (SNR). This thermal noise will cause an accuracy drop in sensor-fusion-based machine learning models when deployed in production. This paper lists some applications for Canada Deuterium Uranium (CANDU) reactors where such sensors are used and therefore can be impacted by the thermal noise issue if machine learning is utilized. A list of recommendations to help mitigate the issue when building future machine learning models for nuclear applications based on sensor fusion is provided. Additionally, this paper demonstrates that machine learning algorithms can be impacted differently by the issue, therefore selecting a more resilient model can help in mitigating it.
       


### [A Comprehensive Indoor Environment Dataset from Single-family Houses in the US](https://arxiv.org/abs/2310.03771)

**Authors:**
Sheik Murad Hassan Anik, Xinghua Gao, Na Meng

**Abstract:**
The paper describes a dataset comprising indoor environmental factors such as temperature, humidity, air quality, and noise levels. The data was collected from 10 sensing devices installed in various locations within three single-family houses in Virginia, USA. The objective of the data collection was to study the indoor environmental conditions of the houses over time. The data were collected at a frequency of one record per minute for a year, combining over 2.5 million records. The paper provides actual floor plans with sensor placements to aid researchers and practitioners in creating reliable building performance models. The techniques used to collect and verify the data are also explained in the paper. The resulting dataset can be employed to enhance models for building energy consumption, occupant behavior, predictive maintenance, and other relevant purposes.
       


### [Qualitative and quantitative evaluation of a methodology for the Digital Twin creation of brownfield production systems](https://arxiv.org/abs/2310.04422)

**Authors:**
Dominik Braun, Nasser Jazdi, Wolfgang Schloegl, Michael Weyrich

**Abstract:**
The Digital Twin is a well-known concept of industry 4.0 and is the cyber part of a cyber-physical production system providing several benefits such as virtual commissioning or predictive maintenance. The existing production systems are lacking a Digital Twin which has to be created manually in a time-consuming and error-prone process. Therefore, methods to create digital models of existing production systems and their relations between them were developed. This paper presents the implementation of the methodology for the creation of multi-disciplinary relations and a quantitative and qualitative evaluation of the benefits of the methodology.
       


### [Surrogate modeling for stochastic crack growth processes in structural health monitoring applications](https://arxiv.org/abs/2310.07241)

**Authors:**
Nicholas E. Silionis, Konstantinos N. Anyfantis

**Abstract:**
Fatigue crack growth is one of the most common types of deterioration in metal structures with significant implications on their reliability. Recent advances in Structural Health Monitoring (SHM) have motivated the use of structural response data to predict future crack growth under uncertainty, in order to enable a transition towards predictive maintenance. Accurately representing different sources of uncertainty in stochastic crack growth (SCG) processes is a non-trivial task. The present work builds on previous research on physics-based SCG modeling under both material and load-related uncertainty. The aim here is to construct computationally efficient, probabilistic surrogate models for SCG processes that successfully encode these different sources of uncertainty. An approach inspired by latent variable modeling is employed that utilizes Gaussian Process (GP) regression models to enable the surrogates to be used to generate prior distributions for different Bayesian SHM tasks as the application of interest. Implementation is carried out in a numerical setting and model performance is assessed for two fundamental crack SHM problems; namely crack length monitoring (damage quantification) and crack growth monitoring (damage prognosis).
       


### [Predictive Maintenance Model Based on Anomaly Detection in Induction Motors: A Machine Learning Approach Using Real-Time IoT Data](https://arxiv.org/abs/2310.14949)

**Authors:**
Sergio F. Chevtchenko, Monalisa C. M. dos Santos, Diego M. Vieira, Ricardo L. Mota, Elisson Rocha, Bruna V. Cruz, Danilo Araújo, Ermeson Andrade

**Abstract:**
With the support of Internet of Things (IoT) devices, it is possible to acquire data from degradation phenomena and design data-driven models to perform anomaly detection in industrial equipment. This approach not only identifies potential anomalies but can also serve as a first step toward building predictive maintenance policies. In this work, we demonstrate a novel anomaly detection system on induction motors used in pumps, compressors, fans, and other industrial machines. This work evaluates a combination of pre-processing techniques and machine learning (ML) models with a low computational cost. We use a combination of pre-processing techniques such as Fast Fourier Transform (FFT), Wavelet Transform (WT), and binning, which are well-known approaches for extracting features from raw data. We also aim to guarantee an optimal balance between multiple conflicting parameters, such as anomaly detection rate, false positive rate, and inference speed of the solution. To this end, multiobjective optimization and analysis are performed on the evaluated models. Pareto-optimal solutions are presented to select which models have the best results regarding classification metrics and computational effort. Differently from most works in this field that use publicly available datasets to validate their models, we propose an end-to-end solution combining low-cost and readily available IoT sensors. The approach is validated by acquiring a custom dataset from induction motors. Also, we fuse vibration, temperature, and noise data from these sensors as the input to the proposed ML model. Therefore, we aim to propose a methodology general enough to be applied in different industrial contexts in the future.
       


### [Unknown Health States Recognition With Collective Decision Based Deep Learning Networks In Predictive Maintenance Applications](https://arxiv.org/abs/2310.17670)

**Authors:**
Chuyue Lou, M. Amine Atoui

**Abstract:**
At present, decision making solutions developed based on deep learning (DL) models have received extensive attention in predictive maintenance (PM) applications along with the rapid improvement of computing power. Relying on the superior properties of shared weights and spatial pooling, Convolutional Neural Network (CNN) can learn effective representations of health states from industrial data. Many developed CNN-based schemes, such as advanced CNNs that introduce residual learning and multi-scale learning, have shown good performance in health state recognition tasks under the assumption that all the classes are known. However, these schemes have no ability to deal with new abnormal samples that belong to state classes not part of the training set. In this paper, a collective decision framework for different CNNs is proposed. It is based on a One-vs-Rest network (OVRN) to simultaneously achieve classification of known and unknown health states. OVRN learn state-specific discriminative features and enhance the ability to reject new abnormal samples incorporated to different CNNs. According to the validation results on the public dataset of Tennessee Eastman Process (TEP), the proposed CNN-based decision schemes incorporating OVRN have outstanding recognition ability for samples of unknown heath states, while maintaining satisfactory accuracy on known states. The results show that the new DL framework outperforms conventional CNNs, and the one based on residual and multi-scale learning has the best overall performance.
       


### [OrionBench: Benchmarking Time Series Generative Models in the Service of the End-User](https://arxiv.org/abs/2310.17748)

**Authors:**
Sarah Alnegheimish, Laure Berti-Equille, Kalyan Veeramachaneni

**Abstract:**
Time series anomaly detection is a vital task in many domains, including patient monitoring in healthcare, forecasting in finance, and predictive maintenance in energy industries. This has led to a proliferation of anomaly detection methods, including deep learning-based methods. Benchmarks are essential for comparing the performances of these models as they emerge, in a fair, rigorous, and reproducible approach. Although several benchmarks for comparing models have been proposed, these usually rely on a one-time execution over a limited set of datasets, with comparisons restricted to a few models. We propose OrionBench: an end-user centric, continuously maintained benchmarking framework for unsupervised time series anomaly detection models. Our framework provides universal abstractions to represent models, hyperparameter standardization, extensibility to add new pipelines and datasets, pipeline verification, and frequent releases with published updates of the benchmark. We demonstrate how to use OrionBench, and the performance of pipelines across 17 releases published over the course of four years. We also walk through two real scenarios we experienced with OrionBench that highlight the importance of continuous benchmarking for unsupervised time series anomaly detection.
       


### [Remaining useful life prediction of Lithium-ion batteries using spatio-temporal multimodal attention networks](https://arxiv.org/abs/2310.18924)

**Authors:**
Sungho Suh, Dhruv Aditya Mittal, Hymalai Bello, Bo Zhou, Mayank Shekhar Jha, Paul Lukowicz

**Abstract:**
Lithium-ion batteries are widely used in various applications, including electric vehicles and renewable energy storage. The prediction of the remaining useful life (RUL) of batteries is crucial for ensuring reliable and efficient operation, as well as reducing maintenance costs. However, determining the life cycle of batteries in real-world scenarios is challenging, and existing methods have limitations in predicting the number of cycles iteratively. In addition, existing works often oversimplify the datasets, neglecting important features of the batteries such as temperature, internal resistance, and material type. To address these limitations, this paper proposes a two-stage RUL prediction scheme for Lithium-ion batteries using a spatio-temporal multimodal attention network (ST-MAN). The proposed ST-MAN is to capture the complex spatio-temporal dependencies in the battery data, including the features that are often neglected in existing works. Despite operating without prior knowledge of end-of-life (EOL) events, our method consistently achieves lower error rates, boasting mean absolute error (MAE) and mean square error (MSE) of 0.0275 and 0.0014, respectively, compared to existing convolutional neural networks (CNN) and long short-term memory (LSTM)-based methods. The proposed method has the potential to improve the reliability and efficiency of battery operations and is applicable in various industries.
       


## November
### [Reinforcement Twinning: from digital twins to model-based reinforcement learning](https://arxiv.org/abs/2311.03628)

**Authors:**
Lorenzo Schena, Pedro Marques, Romain Poletti, Samuel Ahizi, Jan Van den Berghe, Miguel A. Mendez

**Abstract:**
Digital twins promise to revolutionize engineering by offering new avenues for optimization, control, and predictive maintenance. We propose a novel framework for simultaneously training the digital twin of an engineering system and an associated control agent. The twin's training combines adjoint-based data assimilation and system identification methods, while the control agent's training merges model-based optimal control with model-free reinforcement learning. The control agent evolves along two independent paths: one driven by model-based optimal control and the other by reinforcement learning. The digital twin serves as a virtual environment for confrontation and indirect interaction, functioning as an "expert demonstrator." The best policy is selected for real-world interaction and cloned to the other path if training stagnates. We call this framework Reinforcement Twinning (RT). The framework is tested on three diverse engineering systems and control tasks: (1) controlling a wind turbine under varying wind speeds, (2) trajectory control of flapping-wing micro air vehicles (FWMAVs) facing wind gusts, and (3) mitigating thermal loads in managing cryogenic storage tanks. These test cases use simplified models with known ground truth closure laws. Results show that the adjoint-based digital twin training is highly sample-efficient, completing within a few iterations. For the control agent training, both model-based and model-free approaches benefit from their complementary learning experiences. The promising results pave the way for implementing the RT framework on real systems.
       


### [CNN-Based Structural Damage Detection using Time-Series Sensor Data](https://arxiv.org/abs/2311.04252)

**Authors:**
Ishan Pathak, Ishan Jha, Aditya Sadana, Basuraj Bhowmik

**Abstract:**
Structural Health Monitoring (SHM) is vital for evaluating structural condition, aiming to detect damage through sensor data analysis. It aligns with predictive maintenance in modern industry, minimizing downtime and costs by addressing potential structural issues. Various machine learning techniques have been used to extract valuable information from vibration data, often relying on prior structural knowledge. This research introduces an innovative approach to structural damage detection, utilizing a new Convolutional Neural Network (CNN) algorithm. In order to extract deep spatial features from time series data, CNNs are taught to recognize long-term temporal connections. This methodology combines spatial and temporal features, enhancing discrimination capabilities when compared to methods solely reliant on deep spatial features. Time series data are divided into two categories using the proposed neural network: undamaged and damaged. To validate its efficacy, the method's accuracy was tested using a benchmark dataset derived from a three-floor structure at Los Alamos National Laboratory (LANL). The outcomes show that the new CNN algorithm is very accurate in spotting structural degradation in the examined structure.
       


### [State-of-the-art review and synthesis: A requirement-based roadmap for standardized predictive maintenance automation using digital twin technologies](https://arxiv.org/abs/2311.06993)

**Authors:**
Sizhe Ma, Katherine A. Flanigan, Mario Bergés

**Abstract:**
Recent digital advances have popularized predictive maintenance (PMx), offering enhanced efficiency, automation, accuracy, cost savings, and independence in maintenance processes. Yet, PMx continues to face numerous limitations such as poor explainability, sample inefficiency of data-driven methods, complexity of physics-based methods, and limited generalizability and scalability of knowledge-based methods. This paper proposes leveraging Digital Twins (DTs) to address these challenges and enable automated PMx adoption on a larger scale. While DTs have the potential to be transformative, they have not yet reached the maturity needed to bridge these gaps in a standardized manner. Without a standard definition guiding this evolution, the transformation lacks a solid foundation for development. This paper provides a requirement-based roadmap to support standardized PMx automation using DT technologies. Our systematic approach comprises two primary stages. First, we methodically identify the Informational Requirements (IRs) and Functional Requirements (FRs) for PMx, which serve as a foundation from which any unified framework must emerge. Our approach to defining and using IRs and FRs as the backbone of any PMx DT is supported by the proven success of these requirements as blueprints in other areas, such as product development in the software industry. Second, we conduct a thorough literature review across various fields to assess how these IRs and FRs are currently being applied within DTs, enabling us to identify specific areas where further research is needed to support the progress and maturation of requirement-based PMx DTs.
       


### [Strategic Data Augmentation with CTGAN for Smart Manufacturing: Enhancing Machine Learning Predictions of Paper Breaks in Pulp-and-Paper Production](https://arxiv.org/abs/2311.09333)

**Authors:**
Hamed Khosravi, Sarah Farhadpour, Manikanta Grandhi, Ahmed Shoyeb Raihan, Srinjoy Das, Imtiaz Ahmed

**Abstract:**
A significant challenge for predictive maintenance in the pulp-and-paper industry is the infrequency of paper breaks during the production process. In this article, operational data is analyzed from a paper manufacturing machine in which paper breaks are relatively rare but have a high economic impact. Utilizing a dataset comprising 18,398 instances derived from a quality assurance protocol, we address the scarcity of break events (124 cases) that pose a challenge for machine learning predictive models. With the help of Conditional Generative Adversarial Networks (CTGAN) and Synthetic Minority Oversampling Technique (SMOTE), we implement a novel data augmentation framework. This method ensures that the synthetic data mirrors the distribution of the real operational data but also seeks to enhance the performance metrics of predictive modeling. Before and after the data augmentation, we evaluate three different machine learning algorithms-Decision Trees (DT), Random Forest (RF), and Logistic Regression (LR). Utilizing the CTGAN-enhanced dataset, our study achieved significant improvements in predictive maintenance performance metrics. The efficacy of CTGAN in addressing data scarcity was evident, with the models' detection of machine breaks (Class 1) improving by over 30% for Decision Trees, 20% for Random Forest, and nearly 90% for Logistic Regression. With this methodological advancement, this study contributes to industrial quality control and maintenance scheduling by addressing rare event prediction in manufacturing processes.
       


### [Utilizing VQ-VAE for End-to-End Health Indicator Generation in Predicting Rolling Bearing RUL](https://arxiv.org/abs/2311.10525)

**Authors:**
Junliang Wang, Qinghua Zhang, Guanhua Zhu, Guoxi Sun

**Abstract:**
The prediction of the remaining useful life (RUL) of rolling bearings is a pivotal issue in industrial production. A crucial approach to tackling this issue involves transforming vibration signals into health indicators (HI) to aid model training. This paper presents an end-to-end HI construction method, vector quantised variational autoencoder (VQ-VAE), which addresses the need for dimensionality reduction of latent variables in traditional unsupervised learning methods such as autoencoder. Moreover, concerning the inadequacy of traditional statistical metrics in reflecting curve fluctuations accurately, two novel statistical metrics, mean absolute distance (MAD) and mean variance (MV), are introduced. These metrics accurately depict the fluctuation patterns in the curves, thereby indicating the model's accuracy in discerning similar features. On the PMH2012 dataset, methods employing VQ-VAE for label construction achieved lower values for MAD and MV. Furthermore, the ASTCN prediction model trained with VQ-VAE labels demonstrated commendable performance, attaining the lowest values for MAD and MV.
       


### [Utilizing Multiple Inputs Autoregressive Models for Bearing Remaining Useful Life Prediction](https://arxiv.org/abs/2311.16192)

**Authors:**
Junliang Wang, Qinghua Zhang, Guanhua Zhu, Guoxi Sun

**Abstract:**
Accurate prediction of the Remaining Useful Life (RUL) of rolling bearings is crucial in industrial production, yet existing models often struggle with limited generalization capabilities due to their inability to fully process all vibration signal patterns. We introduce a novel multi-input autoregressive model to address this challenge in RUL prediction for bearings. Our approach uniquely integrates vibration signals with previously predicted Health Indicator (HI) values, employing feature fusion to output current window HI values. Through autoregressive iterations, the model attains a global receptive field, effectively overcoming the limitations in generalization. Furthermore, we innovatively incorporate a segmentation method and multiple training iterations to mitigate error accumulation in autoregressive models. Empirical evaluation on the PMH2012 dataset demonstrates that our model, compared to other backbone networks using similar autoregressive approaches, achieves significantly lower Root Mean Square Error (RMSE) and Score. Notably, it outperforms traditional autoregressive models that use label values as inputs and non-autoregressive networks, showing superior generalization abilities with a marked lead in RMSE and Score metrics.
       


## December
### [FaultFormer: Pretraining Transformers for Adaptable Bearing Fault Classification](https://arxiv.org/abs/2312.02380)

**Authors:**
Anthony Zhou, Amir Barati Farimani

**Abstract:**
The growth of global consumption has motivated important applications of deep learning to smart manufacturing and machine health monitoring. In particular, analyzing vibration data offers great potential to extract meaningful insights into predictive maintenance by the detection of bearing faults. Deep learning can be a powerful method to predict these mechanical failures; however, they lack generalizability to new tasks or datasets and require expensive, labeled mechanical data. We address this by presenting a novel self-supervised pretraining and fine-tuning framework based on transformer models. In particular, we investigate different tokenization and data augmentation strategies to reach state-of-the-art accuracies using transformer models. Furthermore, we demonstrate self-supervised masked pretraining for vibration signals and its application to low-data regimes, task adaptation, and dataset adaptation. Pretraining is able to improve performance on scarce, unseen training samples, as well as when fine-tuning on fault classes outside of the pretraining distribution. Furthermore, pretrained transformers are shown to be able to generalize to a different dataset in a few-shot manner. This introduces a new paradigm where models can be pretrained on unlabeled data from different bearings, faults, and machinery and quickly deployed to new, data-scarce applications to suit specific manufacturing needs.
       


### [Semi-Supervised Health Index Monitoring with Feature Generation and Fusion](https://arxiv.org/abs/2312.02867)

**Authors:**
Gaëtan Frusque, Ismail Nejjar, Majid Nabavi, Olga Fink

**Abstract:**
The Health Index (HI) is crucial for evaluating system health and is important for tasks like anomaly detection and Remaining Useful Life (RUL) prediction of safety-critical systems. Real-time, meticulous monitoring of system conditions is essential, especially in manufacturing high-quality and safety-critical components such as spray coatings. However, acquiring accurate health status information (HI labels) in real scenarios can be difficult or costly because it requires continuous, precise measurements that fully capture the system's health. As a result, using datasets from systems run-to-failure, which provide limited HI labels only at the healthy and end-of-life phases, becomes a practical approach. We employ Deep Semi-supervised Anomaly Detection (DeepSAD) embeddings to tackle the challenge of extracting features associated with the system's health state. Additionally, we introduce a diversity loss to further enrich the DeepSAD embeddings. We also propose applying an alternating projection algorithm with isotonic constraints to transform the embedding into a normalized HI with an increasing trend. Validation on the PHME2010 milling dataset, a recognized benchmark with ground truth HIs, confirms the efficacy of our proposed HI estimations. Our methodology is further applied to monitor the wear states of thermal spray coatings using high-frequency voltage. These contributions facilitate more accessible and reliable HI estimation, particularly in scenarios where obtaining ground truth HI labels is impossible.
       


### [State of Health Estimation for Battery Modules with Parallel-Connected Cells Under Cell-to-Cell Variations](https://arxiv.org/abs/2312.03097)

**Authors:**
Qinan Zhou, Dyche Anderson, Jing Sun

**Abstract:**
State of health (SOH) estimation for lithium-ion battery modules with cells connected in parallel is a challenging problem, especially with cell-to-cell variations. Incremental capacity analysis (ICA) and differential voltage analysis (DVA) are effective at the cell level, but a generalizable method to extend them to module-level SOH estimation remains missing, when only module-level measurements are available. This paper proposes a new method and demonstrates that, with multiple features systematically selected from the module-level ICA and DVA, the module-level SOH can be estimated with high accuracy and confidence in the presence of cell-to-cell variations. First, an information theory-based feature selection algorithm is proposed to find an optimal set of features for module-level SOH estimation. Second, a relevance vector regression (RVR)-based module-level SOH estimation model is proposed to provide both point estimates and three-sigma credible intervals while maintaining model sparsity. With more selected features incorporated, the proposed method achieves better estimation accuracy and higher confidence at the expense of higher model complexity. When applied to a large experimental dataset, the proposed method and the resulting sparse model lead to module-level SOH estimates with a 0.5% root-mean-square error and a 1.5% average three-sigma value. With all the training processes completed offboard, the proposed method has low computational complexity for onboard implementations.
       


### [PhysioCHI: Towards Best Practices for Integrating Physiological Signals in HCI](https://arxiv.org/abs/2312.04223)

**Authors:**
Francesco Chiossi, Ekaterina R. Stepanova, Benjamin Tag, Monica Perusquia-Hernandez, Alexandra Kitson, Arindam Dey, Sven Mayer, Abdallah El Ali

**Abstract:**
Recently, we saw a trend toward using physiological signals in interactive systems. These signals, offering deep insights into users' internal states and health, herald a new era for HCI. However, as this is an interdisciplinary approach, many challenges arise for HCI researchers, such as merging diverse disciplines, from understanding physiological functions to design expertise. Also, isolated research endeavors limit the scope and reach of findings. This workshop aims to bridge these gaps, fostering cross-disciplinary discussions on usability, open science, and ethics tied to physiological data in HCI. In this workshop, we will discuss best practices for embedding physiological signals in interactive systems. Through collective efforts, we seek to craft a guiding document for best practices in physiological HCI research, ensuring that it remains grounded in shared principles and methodologies as the field advances.
       


### [D3A-TS: Denoising-Driven Data Augmentation in Time Series](https://arxiv.org/abs/2312.05550)

**Authors:**
David Solis-Martin, Juan Galan-Paez, Joaquin Borrego-Diaz

**Abstract:**
It has been demonstrated that the amount of data is crucial in data-driven machine learning methods. Data is always valuable, but in some tasks, it is almost like gold. This occurs in engineering areas where data is scarce or very expensive to obtain, such as predictive maintenance, where faults are rare. In this context, a mechanism to generate synthetic data can be very useful. While in fields such as Computer Vision or Natural Language Processing synthetic data generation has been extensively explored with promising results, in other domains such as time series it has received less attention. This work specifically focuses on studying and analyzing the use of different techniques for data augmentation in time series for classification and regression problems. The proposed approach involves the use of diffusion probabilistic models, which have recently achieved successful results in the field of Image Processing, for data augmentation in time series. Additionally, the use of meta-attributes to condition the data augmentation process is investigated. The results highlight the high utility of this methodology in creating synthetic data to train classification and regression models. To assess the results, six different datasets from diverse domains were employed, showcasing versatility in terms of input size and output types. Finally, an extensive ablation study is conducted to further support the obtained outcomes.
       


### [Forecasting Lithium-Ion Battery Longevity with Limited Data Availability: Benchmarking Different Machine Learning Algorithms](https://arxiv.org/abs/2312.05717)

**Authors:**
Hudson Hilal, Pramit Saha

**Abstract:**
As the use of Lithium-ion batteries continues to grow, it becomes increasingly important to be able to predict their remaining useful life. This work aims to compare the relative performance of different machine learning algorithms, both traditional machine learning and deep learning, in order to determine the best-performing algorithms for battery cycle life prediction based on minimal data. We investigated 14 different machine learning models that were fed handcrafted features based on statistical data and split into 3 feature groups for testing. For deep learning models, we tested a variety of neural network models including different configurations of standard Recurrent Neural Networks, Gated Recurrent Units, and Long Short Term Memory with and without attention mechanism. Deep learning models were fed multivariate time series signals based on the raw data for each battery across the first 100 cycles. Our experiments revealed that the machine learning algorithms on handcrafted features performed particularly well, resulting in 10-20% average mean absolute percentage error. The best-performing algorithm was the Random Forest Regressor, which gave a minimum 9.8% mean absolute percentage error. Traditional machine learning models excelled due to their capability to comprehend general data set trends. In comparison, deep learning models were observed to perform particularly poorly on raw, limited data. Algorithms like GRU and RNNs that focused on capturing medium-range data dependencies were less adept at recognizing the gradual, slow trends critical for this task. Our investigation reveals that implementing machine learning models with hand-crafted features proves to be more effective than advanced deep learning models for predicting the remaining useful Lithium-ion battery life with limited data availability.
       


### [Do Bayesian Neural Networks Improve Weapon System Predictive Maintenance?](https://arxiv.org/abs/2312.10494)

**Authors:**
Michael Potter, Miru Jun

**Abstract:**
We implement a Bayesian inference process for Neural Networks to model the time to failure of highly reliable weapon systems with interval-censored data and time-varying covariates. We analyze and benchmark our approach, LaplaceNN, on synthetic and real datasets with standard classification metrics such as Receiver Operating Characteristic (ROC) Area Under Curve (AUC) Precision-Recall (PR) AUC, and reliability curve visualizations.
       


### [Exploring Sound vs Vibration for Robust Fault Detection on Rotating Machinery](https://arxiv.org/abs/2312.10742)

**Authors:**
Serkan Kiranyaz, Ozer Can Devecioglu, Amir Alhams, Sadok Sassi, Turker Ince, Onur Avci, Moncef Gabbouj

**Abstract:**
Robust and real-time detection of faults on rotating machinery has become an ultimate objective for predictive maintenance in various industries. Vibration-based Deep Learning (DL) methodologies have become the de facto standard for bearing fault detection as they can produce state-of-the-art detection performances under certain conditions. Despite such particular focus on the vibration signal, the utilization of sound, on the other hand, has been neglected whilst only a few studies have been proposed during the last two decades, all of which were based on a conventional ML approach. One major reason is the lack of a benchmark dataset providing a large volume of both vibration and sound data over several working conditions for different machines and sensor locations. In this study, we address this need by presenting the new benchmark Qatar University Dual-Machine Bearing Fault Benchmark dataset (QU-DMBF), which encapsulates sound and vibration data from two different motors operating under 1080 working conditions overall. Then we draw the focus on the major limitations and drawbacks of vibration-based fault detection due to numerous installation and operational conditions. Finally, we propose the first DL approach for sound-based fault detection and perform comparative evaluations between the sound and vibration over the QU-DMBF dataset. A wide range of experimental results shows that the sound-based fault detection method is significantly more robust than its vibration-based counterpart, as it is entirely independent of the sensor location, cost-effective (requiring no sensor and sensor maintenance), and can achieve the same level of the best detection performance by its vibration-based counterpart. With this study, the QU-DMBF dataset, the optimized source codes in PyTorch, and comparative evaluations are now publicly shared.
       


### [Battery-Care Resource Allocation and Task Offloading in Multi-Agent Post-Disaster MEC Environment](https://arxiv.org/abs/2312.15380)

**Authors:**
Yiwei Tang, Hualong Huang, Wenhan Zhan, Geyong Min, Zhekai Duan, Yuchuan Lei

**Abstract:**
Being an up-and-coming application scenario of mobile edge computing (MEC), the post-disaster rescue suffers multitudinous computing-intensive tasks but unstably guaranteed network connectivity. In rescue environments, quality of service (QoS), such as task execution delay, energy consumption and battery state of health (SoH), is of significant meaning. This paper studies a multi-user post-disaster MEC environment with unstable 5G communication, where device-to-device (D2D) link communication and dynamic voltage and frequency scaling (DVFS) are adopted to balance each user's requirement for task delay and energy consumption. A battery degradation evaluation approach to prolong battery lifetime is also presented. The distributed optimization problem is formulated into a mixed cooperative-competitive (MCC) multi-agent Markov decision process (MAMDP) and is tackled with recurrent multi-agent Proximal Policy Optimization (rMAPPO). Extensive simulations and comprehensive comparisons with other representative algorithms clearly demonstrate the effectiveness of the proposed rMAPPO-based offloading scheme.
       


### [PINN surrogate of Li-ion battery models for parameter inference. Part I: Implementation and multi-fidelity hierarchies for the single-particle model](https://arxiv.org/abs/2312.17329)

**Authors:**
Malik Hassanaly, Peter J. Weddle, Ryan N. King, Subhayan De, Alireza Doostan, Corey R. Randall, Eric J. Dufek, Andrew M. Colclasure, Kandler Smith

**Abstract:**
To plan and optimize energy storage demands that account for Li-ion battery aging dynamics, techniques need to be developed to diagnose battery internal states accurately and rapidly. This study seeks to reduce the computational resources needed to determine a battery's internal states by replacing physics-based Li-ion battery models -- such as the single-particle model (SPM) and the pseudo-2D (P2D) model -- with a physics-informed neural network (PINN) surrogate. The surrogate model makes high-throughput techniques, such as Bayesian calibration, tractable to determine battery internal parameters from voltage responses. This manuscript is the first of a two-part series that introduces PINN surrogates of Li-ion battery models for parameter inference (i.e., state-of-health diagnostics). In this first part, a method is presented for constructing a PINN surrogate of the SPM. A multi-fidelity hierarchical training, where several neural nets are trained with multiple physics-loss fidelities is shown to significantly improve the surrogate accuracy when only training on the governing equation residuals. The implementation is made available in a companion repository (https://github.com/NREL/pinnstripes). The techniques used to develop a PINN surrogate of the SPM are extended in Part II for the PINN surrogate for the P2D battery model, and explore the Bayesian calibration capabilities of both surrogates.
       


### [PINN surrogate of Li-ion battery models for parameter inference. Part II: Regularization and application of the pseudo-2D model](https://arxiv.org/abs/2312.17336)

**Authors:**
Malik Hassanaly, Peter J. Weddle, Ryan N. King, Subhayan De, Alireza Doostan, Corey R. Randall, Eric J. Dufek, Andrew M. Colclasure, Kandler Smith

**Abstract:**
Bayesian parameter inference is useful to improve Li-ion battery diagnostics and can help formulate battery aging models. However, it is computationally intensive and cannot be easily repeated for multiple cycles, multiple operating conditions, or multiple replicate cells. To reduce the computational cost of Bayesian calibration, numerical solvers for physics-based models can be replaced with faster surrogates. A physics-informed neural network (PINN) is developed as a surrogate for the pseudo-2D (P2D) battery model calibration. For the P2D surrogate, additional training regularization was needed as compared to the PINN single-particle model (SPM) developed in Part I. Both the PINN SPM and P2D surrogate models are exercised for parameter inference and compared to data obtained from a direct numerical solution of the governing equations. A parameter inference study highlights the ability to use these PINNs to calibrate scaling parameters for the cathode Li diffusion and the anode exchange current density. By realizing computational speed-ups of 2250x for the P2D model, as compared to using standard integrating methods, the PINN surrogates enable rapid state-of-health diagnostics. In the low-data availability scenario, the testing error was estimated to 2mV for the SPM surrogate and 10mV for the P2D surrogate which could be mitigated with additional data.
       


# 2024
## January
### [Utilizing Autoregressive Networks for Full Lifecycle Data Generation of Rolling Bearings for RUL Prediction](https://arxiv.org/abs/2401.01119)

**Authors:**
Junliang Wang, Qinghua Zhang, Guanhua Zhu, Guoxi Sun

**Abstract:**
The prediction of rolling bearing lifespan is of significant importance in industrial production. However, the scarcity of high-quality, full lifecycle data has been a major constraint in achieving precise predictions. To address this challenge, this paper introduces the CVGAN model, a novel framework capable of generating one-dimensional vibration signals in both horizontal and vertical directions, conditioned on historical vibration data and remaining useful life. In addition, we propose an autoregressive generation method that can iteratively utilize previously generated vibration information to guide the generation of current signals. The effectiveness of the CVGAN model is validated through experiments conducted on the PHM 2012 dataset. Our findings demonstrate that the CVGAN model, in terms of both MMD and FID metrics, outperforms many advanced methods in both autoregressive and non-autoregressive generation modes. Notably, training using the full lifecycle data generated by the CVGAN model significantly improves the performance of the predictive model. This result highlights the effectiveness of the data generated by CVGans in enhancing the predictive power of these models.
       


### [Image-based Deep Learning for Smart Digital Twins: a Review](https://arxiv.org/abs/2401.02523)

**Authors:**
Md Ruman Islam, Mahadevan Subramaniam, Pei-Chi Huang

**Abstract:**
Smart Digital twins (SDTs) are being increasingly used to virtually replicate and predict the behaviors of complex physical systems through continual data assimilation enabling the optimization of the performance of these systems by controlling the actions of systems. Recently, deep learning (DL) models have significantly enhanced the capabilities of SDTs, particularly for tasks such as predictive maintenance, anomaly detection, and optimization. In many domains, including medicine, engineering, and education, SDTs use image data (image-based SDTs) to observe and learn system behaviors and control their behaviors. This paper focuses on various approaches and associated challenges in developing image-based SDTs by continually assimilating image data from physical systems. The paper also discusses the challenges involved in designing and implementing DL models for SDTs, including data acquisition, processing, and interpretation. In addition, insights into the future directions and opportunities for developing new image-based DL approaches to develop robust SDTs are provided. This includes the potential for using generative models for data augmentation, developing multi-modal DL models, and exploring the integration of DL with other technologies, including 5G, edge computing, and IoT. In this paper, we describe the image-based SDTs, which enable broader adoption of the digital twin DT paradigms across a broad spectrum of areas and the development of new methods to improve the abilities of SDTs in replicating, predicting, and optimizing the behavior of complex systems.
       


### [A Change Point Detection Integrated Remaining Useful Life Estimation Model under Variable Operating Conditions](https://arxiv.org/abs/2401.04351)

**Authors:**
Anushiya Arunan, Yan Qin, Xiaoli Li, Chau Yuen

**Abstract:**
By informing the onset of the degradation process, health status evaluation serves as a significant preliminary step for reliable remaining useful life (RUL) estimation of complex equipment. This paper proposes a novel temporal dynamics learning-based model for detecting change points of individual devices, even under variable operating conditions, and utilises the learnt change points to improve the RUL estimation accuracy. During offline model development, the multivariate sensor data are decomposed to learn fused temporal correlation features that are generalisable and representative of normal operation dynamics across multiple operating conditions. Monitoring statistics and control limit thresholds for normal behaviour are dynamically constructed from these learnt temporal features for the unsupervised detection of device-level change points. The detected change points then inform the degradation data labelling for training a long short-term memory (LSTM)-based RUL estimation model. During online monitoring, the temporal correlation dynamics of a query device is monitored for breach of the control limit derived in offline training. If a change point is detected, the device's RUL is estimated with the well-trained offline model for early preventive action. Using C-MAPSS turbofan engines as the case study, the proposed method improved the accuracy by 5.6\% and 7.5\% for two scenarios with six operating conditions, when compared to existing LSTM-based RUL estimation models that do not consider heterogeneous change points.
       


### [Towards a BMS2 Design Framework: Adaptive Data-driven State-of-health Estimation for Second-Life Batteries with BIBO Stability Guarantees](https://arxiv.org/abs/2401.04734)

**Authors:**
Xiaofan Cui, Muhammad Aadil Khan, Surinder Singh, Ratnesh Sharma, Simona Onori

**Abstract:**
A key challenge that is currently hindering the widespread use of retired electric vehicle (EV) batteries for second-life (SL) applications is the ability to accurately estimate and monitor their state of health (SOH). Second-life battery systems can be sourced from different battery packs with lack of knowledge of their historical usage. To tackle the in-the-field use of SL batteries, this paper introduces an online adaptive health estimation approach with guaranteed bounded-input-bounded-output (BIBO) stability. This method relies exclusively on operational data that can be accessed in real time from SL batteries. The effectiveness of the proposed approach is shown on a laboratory aged experimental data set of retired EV batteries. The estimator gains are dynamically adapted to accommodate the distinct characteristics of each individual cell, making it a promising candidate for future SL battery management systems (BMS2).
       


### [Model-Driven Dataset Generation for Data-Driven Battery SOH Models](https://arxiv.org/abs/2401.05474)

**Authors:**
Khaled Sidahmed Sidahmed Alamin, Francesco Daghero, Giovanni Pollo, Daniele Jahier Pagliari, Yukai Chen, Enrico Macii, Massimo Poncino, Sara Vinco

**Abstract:**
Estimating the State of Health (SOH) of batteries is crucial for ensuring the reliable operation of battery systems. Since there is no practical way to instantaneously measure it at run time, a model is required for its estimation. Recently, several data-driven SOH models have been proposed, whose accuracy heavily relies on the quality of the datasets used for their training. Since these datasets are obtained from measurements, they are limited in the variety of the charge/discharge profiles. To address this scarcity issue, we propose generating datasets by simulating a traditional battery model (e.g., a circuit-equivalent one). The primary advantage of this approach is the ability to use a simulatable battery model to evaluate a potentially infinite number of workload profiles for training the data-driven model. Furthermore, this general concept can be applied using any simulatable battery model, providing a fine spectrum of accuracy/complexity tradeoffs. Our results indicate that using simulated data achieves reasonable accuracy in SOH estimation, with a 7.2% error relative to the simulated model, in exchange for a 27X memory reduction and a =2000X speedup.
       


### [Surrogate Neural Networks Local Stability for Aircraft Predictive Maintenance](https://arxiv.org/abs/2401.06821)

**Authors:**
Mélanie Ducoffe, Guillaume Povéda, Audrey Galametz, Ryma Boumazouza, Marion-Cécile Martin, Julien Baris, Derk Daverschot, Eugene O'Higgins

**Abstract:**
Surrogate Neural Networks are nowadays routinely used in industry as substitutes for computationally demanding engineering simulations (e.g., in structural analysis). They allow to generate faster predictions and thus analyses in industrial applications e.g., during a product design, testing or monitoring phases. Due to their performance and time-efficiency, these surrogate models are now being developed for use in safety-critical applications. Neural network verification and in particular the assessment of their robustness (e.g., to perturbations) is the next critical step to allow their inclusion in real-life applications and certification. We assess the applicability and scalability of empirical and formal methods in the context of aircraft predictive maintenance for surrogate neural networks designed to predict the stress sustained by an aircraft part from external loads. The case study covers a high-dimensional input and output space and the verification process thus accommodates multi-objective constraints. We explore the complementarity of verification methods in assessing the local stability property of such surrogate models to input noise. We showcase the effectiveness of sequentially combining methods in one verification 'pipeline' and demonstrate the subsequent gain in runtime required to assess the targeted property.
       


### [Remaining Useful Life Prediction for Aircraft Engines using LSTM](https://arxiv.org/abs/2401.07590)

**Authors:**
Anees Peringal, Mohammed Basheer Mohiuddin, Ahmed Hassan

**Abstract:**
This study uses a Long Short-Term Memory (LSTM) network to predict the remaining useful life (RUL) of jet engines from time-series data, crucial for aircraft maintenance and safety. The LSTM model's performance is compared with a Multilayer Perceptron (MLP) on the C-MAPSS dataset from NASA, which contains jet engine run-to-failure events. The LSTM learns from temporal sequences of sensor data, while the MLP learns from static data snapshots. The LSTM model consistently outperforms the MLP in prediction accuracy, demonstrating its superior ability to capture temporal dependencies in jet engine degradation patterns. The software for this project is in https://github.com/AneesPeringal/rul-prediction.git.
       


### [Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities](https://arxiv.org/abs/2401.07871)

**Authors:**
Logan Cummins, Alex Sommers, Somayeh Bakhtiari Ramezani, Sudip Mittal, Joseph Jabour, Maria Seale, Shahram Rahimi

**Abstract:**
Predictive maintenance is a well studied collection of techniques that aims to prolong the life of a mechanical system by using artificial intelligence and machine learning to predict the optimal time to perform maintenance. The methods allow maintainers of systems and hardware to reduce financial and time costs of upkeep. As these methods are adopted for more serious and potentially life-threatening applications, the human operators need trust the predictive system. This attracts the field of Explainable AI (XAI) to introduce explainability and interpretability into the predictive system. XAI brings methods to the field of predictive maintenance that can amplify trust in the users while maintaining well-performing systems. This survey on explainable predictive maintenance (XPM) discusses and presents the current methods of XAI as applied to predictive maintenance while following the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines. We categorize the different XPM methods into groups that follow the XAI literature. Additionally, we include current challenges and a discussion on future research directions in XPM.
       


### [Bias-Compensated State of Charge and State of Health Joint Estimation for Lithium Iron Phosphate Batteries](https://arxiv.org/abs/2401.08136)

**Authors:**
Baozhao Yi, Xinhao Du, Jiawei Zhang, Xiaogang Wu, Qiuhao Hu, Weiran Jiang, Xiaosong Hu, Ziyou Song

**Abstract:**
Accurate estimation of the state of charge (SOC) and state of health (SOH) is crucial for the safe and reliable operation of batteries. Voltage measurement bias highly affects state estimation accuracy, especially in Lithium Iron Phosphate (LFP) batteries, which are susceptible due to their flat open-circuit voltage (OCV) curves. This work introduces a bias-compensated algorithm to reliably estimate the SOC and SOH of LFP batteries under the influence of voltage measurement bias. Specifically, SOC and SOH are estimated using the Dual Extended Kalman Filter (DEKF) in the high-slope SOC range, where voltage measurement bias effects are weak. Besides, the voltage measurement biases estimated in the low-slope SOC regions are compensated in the following joint estimation of SOC and SOH to enhance the state estimation accuracy further. Experimental results indicate that the proposed algorithm significantly outperforms the traditional method, which does not consider biases under different temperatures and aging conditions. Additionally, the bias-compensated algorithm can achieve low estimation errors of below 1.5% for SOC and 2% for SOH, even with a 30mV voltage measurement bias. Finally, even if the voltage measurement biases change in operation, the proposed algorithm can remain robust and keep the estimated errors of states around 2%.
       


### [Design & Implementation of Automatic Machine Condition Monitoring and Maintenance System in Limited Resource Situations](https://arxiv.org/abs/2401.15088)

**Authors:**
Abu Hanif Md. Ripon, Muhammad Ahsan Ullah, Arindam Kumar Paul, Md. Mortaza Morshed

**Abstract:**
In the era of the fourth industrial revolution, it is essential to automate fault detection and diagnosis of machineries so that a warning system can be developed that will help to take an appropriate action before any catastrophic damage. Some machines health monitoring systems are used globally but they are expensive and need trained personnel to operate and analyse. Predictive maintenance and occupational health and safety culture are not available due to inadequate infrastructure, lack of skilled manpower, financial crisis, and others in developing countries. Starting from developing a cost-effective DAS for collecting fault data in this study, the effect of limited data and resources has been investigated while automating the process. To solve this problem, A feature engineering and data reduction method has been developed combining the concepts from wavelets, differential calculus, and signal processing. Finally, for automating the whole process, all the necessary theoretical and practical considerations to develop a predictive model have been proposed. The DAS successfully collected the required data from the machine that is 89% accurate compared to the professional manual monitoring system. SVM and NN were proposed for the prediction purpose because of their high predicting accuracy greater than 95% during training and 100% during testing the new samples. In this study, the combination of the simple algorithm with a rule-based system instead of a data-intensive system turned out to be hybridization by validating with collected data. The outcome of this research can be instantly applied to small and medium-sized industries for finding other issues and developing accordingly. As one of the foundational studies in automatic FDD, the findings and procedure of this study can lead others to extend, generalize, or add other dimensions to FDD automation.
       


### [SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance](https://arxiv.org/abs/2401.15199)

**Authors:**
Zahra Kharazian, Tony Lindgren, Sindri Magnússon, Olof Steinert, Oskar Andersson Reyna

**Abstract:**
This paper presents a description of a real-world, multivariate time series dataset collected from an anonymized engine component (called Component X) of a fleet of trucks from SCANIA, Sweden. This dataset includes diverse variables capturing detailed operational data, repair records, and specifications of trucks while maintaining confidentiality by anonymization. It is well-suited for a range of machine learning applications, such as classification, regression, survival analysis, and anomaly detection, particularly when applied to predictive maintenance scenarios. The large population size and variety of features in the format of histograms and numerical counters, along with the inclusion of temporal information, make this real-world dataset unique in the field. The objective of releasing this dataset is to give a broad range of researchers the possibility of working with real-world data from an internationally well-known company and introduce a standard benchmark to the predictive maintenance field, fostering reproducible research.
       


### [Spatio-Temporal Attention Graph Neural Network for Remaining Useful Life Prediction](https://arxiv.org/abs/2401.15964)

**Authors:**
Zhixin Huang, Yujiang He, Bernhard Sick

**Abstract:**
Remaining useful life prediction plays a crucial role in the health management of industrial systems. Given the increasing complexity of systems, data-driven predictive models have attracted significant research interest. Upon reviewing the existing literature, it appears that many studies either do not fully integrate both spatial and temporal features or employ only a single attention mechanism. Furthermore, there seems to be inconsistency in the choice of data normalization methods, particularly concerning operating conditions, which might influence predictive performance. To bridge these observations, this study presents the Spatio-Temporal Attention Graph Neural Network. Our model combines graph neural networks and temporal convolutional neural networks for spatial and temporal feature extraction, respectively. The cascade of these extractors, combined with multi-head attention mechanisms for both spatio-temporal dimensions, aims to improve predictive precision and refine model explainability. Comprehensive experiments were conducted on the C-MAPSS dataset to evaluate the impact of unified versus clustering normalization. The findings suggest that our model performs state-of-the-art results using only the unified normalization. Additionally, when dealing with datasets with multiple operating conditions, cluster normalization enhances the performance of our proposed model by up to 27%.
       


### [Supervised Contrastive Learning based Dual-Mixer Model for Remaining Useful Life Prediction](https://arxiv.org/abs/2401.16462)

**Authors:**
En Fu, Yanyan Hu, Kaixiang Peng, Yuxin Chu

**Abstract:**
The problem of the Remaining Useful Life (RUL) prediction, aiming at providing an accurate estimate of the remaining time from the current predicting moment to the complete failure of the device, has gained significant attention from researchers in recent years. In this paper, to overcome the shortcomings of rigid combination for temporal and spatial features in most existing RUL prediction approaches, a spatial-temporal homogeneous feature extractor, named Dual-Mixer model, is firstly proposed. Flexible layer-wise progressive feature fusion is employed to ensure the homogeneity of spatial-temporal features and enhance the prediction accuracy. Secondly, the Feature Space Global Relationship Invariance (FSGRI) training method is introduced based on supervised contrastive learning. This method maintains the consistency of relationships among sample features with their degradation patterns during model training, simplifying the subsequently regression task in the output layer and improving the model's performance in RUL prediction. Finally, the effectiveness of the proposed method is validated through comparisons with other latest research works on the C-MAPSS dataset. The Dual-Mixer model demonstrates superiority across most metrics, while the FSGRI training method shows an average improvement of 7.00% and 2.41% in RMSE and MAPE, respectively, for all baseline models. Our experiments and model code are publicly available at https://github.com/fuen1590/PhmDeepLearningProjects.
       


### [Dynamical Survival Analysis with Controlled Latent States](https://arxiv.org/abs/2401.17077)

**Authors:**
Linus Bleistein, Van-Tuan Nguyen, Adeline Fermanian, Agathe Guilloux

**Abstract:**
We consider the task of learning individual-specific intensities of counting processes from a set of static variables and irregularly sampled time series. We introduce a novel modelization approach in which the intensity is the solution to a controlled differential equation. We first design a neural estimator by building on neural controlled differential equations. In a second time, we show that our model can be linearized in the signature space under sufficient regularity conditions, yielding a signature-based estimator which we call CoxSig. We provide theoretical learning guarantees for both estimators, before showcasing the performance of our models on a vast array of simulated and real-world datasets from finance, predictive maintenance and food supply chain management.
       


## February
### [Optimized Task Assignment and Predictive Maintenance for Industrial Machines using Markov Decision Process](https://arxiv.org/abs/2402.00042)

**Authors:**
Ali Nasir, Samir Mekid, Zaid Sawlan, Omar Alsawafy

**Abstract:**
This paper considers a distributed decision-making approach for manufacturing task assignment and condition-based machine health maintenance. Our approach considers information sharing between the task assignment and health management decision-making agents. We propose the design of the decision-making agents based on Markov decision processes. The key advantage of using a Markov decision process-based approach is the incorporation of uncertainty involved in the decision-making process. The paper provides detailed mathematical models along with the associated practical execution strategy. In order to demonstrate the effectiveness and practical applicability of our proposed approach, we have included a detailed numerical case study that is based on open source milling machine tool degradation data. Our case study indicates that the proposed approach offers flexibility in terms of the selection of cost parameters and it allows for offline computation and analysis of the decision-making policy. These features create and opportunity for the future work on learning of the cost parameters associated with our proposed model using artificial intelligence.
       


### [Adapting Amidst Degradation: Cross Domain Li-ion Battery Health Estimation via Physics-Guided Test-Time Training](https://arxiv.org/abs/2402.00068)

**Authors:**
Yuyuan Feng, Guosheng Hu, Xiaodong Li, Zhihong Zhang

**Abstract:**
Health modeling of lithium-ion batteries (LIBs) is crucial for safe and efficient energy management and carries significant socio-economic implications. Although Machine Learning (ML)-based State of Health (SOH) estimation methods have made significant progress in accuracy, the scarcity of high-quality LIB data remains a major obstacle. Existing transfer learning methods for cross-domain LIB SOH estimation have significantly alleviated the labeling burden of target LIB data, however, they still require sufficient unlabeled target data (UTD) for effective adaptation to the target domain. Collecting this UTD is challenging due to the time-consuming nature of degradation experiments. To address this issue, we introduce a practical Test-Time Training framework, BatteryTTT, which adapts the model continually using each UTD collected amidst degradation, thereby significantly reducing data collection time. To fully utilize each UTD, BatteryTTT integrates the inherent physical laws of modern LIBs into self-supervised learning, termed Physcics-Guided Test-Time Training. Additionally, we explore the potential of large language models (LLMs) in battery sequence modeling by evaluating their performance in SOH estimation through model reprogramming and prefix prompt adaptation. The combination of BatteryTTT and LLM modeling, termed GPT4Battery, achieves state-of-the-art generalization results across current LIB benchmarks. Furthermore, we demonstrate the practical value and scalability of our approach by deploying it in our real-world battery management system (BMS) for 300Ah large-scale energy storage LIBs.
       


### [Signal Quality Auditing for Time-series Data](https://arxiv.org/abs/2402.00803)

**Authors:**
Chufan Gao, Nicholas Gisolfi, Artur Dubrawski

**Abstract:**
Signal quality assessment (SQA) is required for monitoring the reliability of data acquisition systems, especially in AI-driven Predictive Maintenance (PMx) application contexts. SQA is vital for addressing "silent failures" of data acquisition hardware and software, which when unnoticed, misinform the users of data, creating the risk for incorrect decisions with unintended or even catastrophic consequences. We have developed an open-source software implementation of signal quality indices (SQIs) for the analysis of time-series data. We codify a range of SQIs, demonstrate them using established benchmark data, and show that they can be effective for signal quality assessment. We also study alternative approaches to denoising time-series data in an attempt to improve the quality of the already degraded signal, and evaluate them empirically on relevant real-world data. To our knowledge, our software toolkit is the first to provide an open source implementation of a broad range of signal quality assessment and improvement techniques validated on publicly available benchmark data for ease of reproducibility. The generality of our framework can be easily extended to assessing reliability of arbitrary time-series measurements in complex systems, especially when morphological patterns of the waveform shapes and signal periodicity are of key interest in downstream analyses.
       


### [Bayesian Deep Learning for Remaining Useful Life Estimation via Stein Variational Gradient Descent](https://arxiv.org/abs/2402.01098)

**Authors:**
Luca Della Libera, Jacopo Andreoli, Davide Dalle Pezze, Mirco Ravanelli, Gian Antonio Susto

**Abstract:**
A crucial task in predictive maintenance is estimating the remaining useful life of physical systems. In the last decade, deep learning has improved considerably upon traditional model-based and statistical approaches in terms of predictive performance. However, in order to optimally plan maintenance operations, it is also important to quantify the uncertainty inherent to the predictions. This issue can be addressed by turning standard frequentist neural networks into Bayesian neural networks, which are naturally capable of providing confidence intervals around the estimates. Several methods exist for training those models. Researchers have focused mostly on parametric variational inference and sampling-based techniques, which notoriously suffer from limited approximation power and large computational burden, respectively. In this work, we use Stein variational gradient descent, a recently proposed algorithm for approximating intractable distributions that overcomes the drawbacks of the aforementioned techniques. In particular, we show through experimental studies on simulated run-to-failure turbofan engine degradation data that Bayesian deep learning models trained via Stein variational gradient descent consistently outperform with respect to convergence speed and predictive performance both the same models trained via parametric variational inference and their frequentist counterparts trained via backpropagation. Furthermore, we propose a method to enhance performance based on the uncertainty information provided by the Bayesian models. We release the source code at https://github.com/lucadellalib/bdl-rul-svgd.
       


### [Novel Low-Complexity Model Development for Li-ion Cells Using Online Impedance Measurement](https://arxiv.org/abs/2402.07777)

**Authors:**
Abhijit Kulkarni, Ahsan Nadeem, Roberta Di Fonso, Yusheng Zheng, Remus Teodorescu

**Abstract:**
Modeling of Li-ion cells is used in battery management systems (BMS) to determine key states such as state-of-charge (SoC), state-of-health (SoH), etc. Accurate models are also useful in developing a cell-level digital-twin that can be used for protection and diagnostics in the BMS. In this paper, a low-complexity model development is proposed based on the equivalent circuit model (ECM) of the Li-ion cells. The proposed approach uses online impedance measurement at discrete frequencies to derive the ECM that matches closely with the results from the electro-impedance spectroscopy (EIS). The proposed method is suitable to be implemented in a microcontroller with low-computational power, typically used in BMS. Practical design guidelines are proposed to ensure fast and accurate model development. Using the proposed method to enhance the functions of a typical automotive BMS is described. Experimental validation is performed using large prismatic cells and small-capacity cylindrical cells. Root-mean-square error (RMSE) of less than 3\% is observed for a wide variation of operating conditions.
       


### [Advancements in Point Cloud-Based 3D Defect Detection and Classification for Industrial Systems: A Comprehensive Survey](https://arxiv.org/abs/2402.12923)

**Authors:**
Anju Rani, Daniel Ortiz-Arroyo, Petar Durdevic

**Abstract:**
In recent years, 3D point clouds (PCs) have gained significant attention due to their diverse applications across various fields, such as computer vision (CV), condition monitoring (CM), virtual reality, robotics, autonomous driving, etc. Deep learning (DL) has proven effective in leveraging 3D PCs to address various challenges encountered in 2D vision. However, applying deep neural networks (DNNs) to process 3D PCs presents unique challenges. This paper provides an in-depth review of recent advancements in DL-based industrial CM using 3D PCs, with a specific focus on defect shape classification and segmentation within industrial applications. Recognizing the crucial role of these aspects in industrial maintenance, the paper offers insightful observations on the strengths and limitations of the reviewed DL-based PC processing methods. This knowledge synthesis aims to contribute to understanding and enhancing CM processes, particularly within the framework of remaining useful life (RUL), in industrial systems.
       


## March
### [Frailty or Frailties: Exploring Frailty Index Subdimensions in the English Longitudinal Study of Ageing](https://arxiv.org/abs/2403.00472)

**Authors:**
Lara Johnson, Bruce Guthrie, Paul A T Kelly, Atul Anand, Alan Marshall, Sohan Seth

**Abstract:**
Background: Frailty, a state of increased vulnerability to adverse health outcomes, has garnered significant attention in research and clinical practice. Existing constructs aggregate clinical features or health deficits into a single score. While simple and interpretable, this approach may overlook the complexity of frailty and not capture the full range of variation between individuals.
  Methods: Exploratory factor analysis was used to infer latent dimensions of a frailty index constructed using survey data from the English Longitudinal Study of Ageing (ELSA), wave 9. The dataset included 58 self-reported health deficits in a representative sample of community-dwelling adults aged 65+ (N = 4971). Deficits encompassed chronic disease, general health status, mobility, independence with activities of daily living, psychological wellbeing, memory and cognition. Multiple linear regression examined associations with CASP-19 quality of life scores.
  Results: Factor analysis revealed four frailty subdimensions. Based on the component deficits with the highest loading values, these factors were labelled "Mobility Impairment and Physical Morbidity", "Difficulties in Daily Activities", "Mental Health" and "Disorientation in Time". The four subdimensions were a better predictor of quality of life than frailty index scores.
  Conclusions: Distinct subdimensions of frailty can be identified from standard index scores. A decomposed approach to understanding frailty has potential to provide a more nuanced understanding of an individual's state of health across multiple deficits.
       


### [MambaLithium: Selective state space model for remaining-useful-life, state-of-health, and state-of-charge estimation of lithium-ion batteries](https://arxiv.org/abs/2403.05430)

**Author:**
Zhuangwei Shi

**Abstract:**
Recently, lithium-ion batteries occupy a pivotal position in the realm of electric vehicles and the burgeoning new energy industry. Their performance is heavily dependent on three core states: remaining-useful-life (RUL), state-of-health (SOH), and state-of-charge (SOC). Given the remarkable success of Mamba (Structured state space sequence models with selection mechanism and scan module, S6) in sequence modeling tasks, this paper introduces MambaLithium, a selective state space model tailored for precise estimation of these critical battery states. Leveraging Mamba algorithms, MambaLithium adeptly captures the intricate aging and charging dynamics of lithium-ion batteries. By focusing on pivotal states within the battery's operational envelope, MambaLithium not only enhances estimation accuracy but also maintains computational robustness. Experiments conducted using real-world battery data have validated the model's superiority in predicting battery health and performance metrics, surpassing current methods. The proposed MambaLithium framework is potential for applications in advancing battery management systems and fostering sustainable energy storage solutions. Source code is available at https://github.com/zshicode/MambaLithium.
       


### [Micro-Fracture Detection in Photovoltaic Cells with Hardware-Constrained Devices and Computer Vision](https://arxiv.org/abs/2403.05694)

**Authors:**
Booy Vitas Faassen, Jorge Serrano, Paul D. Rosero-Montalvo

**Abstract:**
Solar energy is rapidly becoming a robust renewable energy source to conventional finite resources such as fossil fuels. It is harvested using interconnected photovoltaic panels, typically built with crystalline silicon cells, i.e. semiconducting materials that convert effectively the solar radiation into electricity. However, crystalline silicon is fragile and vulnerable to cracking over time or in predictive maintenance tasks, which can lead to electric isolation of parts of the solar cell and even failure, thus affecting the panel performance and reducing electricity generation. This work aims to developing a system for detecting cell cracks in solar panels to anticipate and alaert of a potential failure of the photovoltaic system by using computer vision techniques. Three scenarios are defined where these techniques will bring value. In scenario A, images are taken manually and the system detecting failures in the solar cells is not subject to any computationa constraints. In scenario B, an Edge device is placed near the solar farm, able to make inferences. Finally, in scenario C, a small microcontroller is placed in a drone flying over the solar farm and making inferences about the solar cells' states. Three different architectures are found the most suitable solutions, one for each scenario, namely the InceptionV3 model, an EfficientNetB0 model shrunk into full integer quantization, and a customized CNN architechture built with VGG16 blocks.
       


### [Experimental Comparison of Ensemble Methods and Time-to-Event Analysis Models Through Integrated Brier Score and Concordance Index](https://arxiv.org/abs/2403.07460)

**Authors:**
Camila Fernandez, Chung Shue Chen, Chen Pierre Gaillard, Alonso Silva

**Abstract:**
Time-to-event analysis is a branch of statistics that has increased in popularity during the last decades due to its many application fields, such as predictive maintenance, customer churn prediction and population lifetime estimation. In this paper, we review and compare the performance of several prediction models for time-to-event analysis. These consist of semi-parametric and parametric statistical models, in addition to machine learning approaches. Our study is carried out on three datasets and evaluated in two different scores (the integrated Brier score and concordance index). Moreover, we show how ensemble methods, which surprisingly have not yet been much studied in time-to-event analysis, can improve the prediction accuracy and enhance the robustness of the prediction performance. We conclude the analysis with a simulation experiment in which we evaluate the factors influencing the performance ranking of the methods using both scores.
       


### [Comprehensive Study Of Predictive Maintenance In Industries Using Classification Models And LSTM Model](https://arxiv.org/abs/2403.10259)

**Authors:**
Saket Maheshwari, Sambhav Tiwari, Shyam Rai, Satyam Vinayak Daman Pratap Singh

**Abstract:**
In today's technology-driven era, the imperative for predictive maintenance and advanced diagnostics extends beyond aviation to encompass the identification of damages, failures, and operational defects in rotating and moving machines. Implementing such services not only curtails maintenance costs but also extends machine lifespan, ensuring heightened operational efficiency. Moreover, it serves as a preventive measure against potential accidents or catastrophic events. The advent of Artificial Intelligence (AI) has revolutionized maintenance across industries, enabling more accurate and efficient prediction and analysis of machine failures, thereby conserving time and resources. Our proposed study aims to delve into various machine learning classification techniques, including Support Vector Machine (SVM), Random Forest, Logistic Regression, and Convolutional Neural Network LSTM-Based, for predicting and analyzing machine performance. SVM classifies data into different categories based on their positions in a multidimensional space, while Random Forest employs ensemble learning to create multiple decision trees for classification. Logistic Regression predicts the probability of binary outcomes using input data. The primary objective of the study is to assess these algorithms' performance in predicting and analyzing machine performance, considering factors such as accuracy, precision, recall, and F1 score. The findings will aid maintenance experts in selecting the most suitable machine learning algorithm for effective prediction and analysis of machine performance.
       


### [Stochastic compartment model with mortality and its application to epidemic spreading in complex networks](https://arxiv.org/abs/2403.11774)

**Authors:**
Teo Granger, Thomas M. Michelitsch, Michael Bestehorn, Alejandro P. Riascos, Bernard A. Collet

**Abstract:**
We study epidemic spreading in complex networks by a multiple random walker approach. Each walker performs an independent simple Markovian random walk on a complex undirected (ergodic) random graph where we focus on Barabási-Albert (BA), Erdös-Rényi (ER) and Watts-Strogatz (WS) types. Both, walkers and nodes can be either susceptible (S) or infected and infectious (I) representing their states of health. Susceptible nodes may be infected by visits of infected walkers, and susceptible walkers may be infected by visiting infected nodes. No direct transmission of the disease among walkers (or among nodes) is possible. This model mimics a large class of diseases such as Dengue and Malaria with transmission of the disease via vectors (mosquitos). Infected walkers may die during the time span of their infection introducing an additional compartment D of dead walkers. Infected nodes never die and always recover from their infection after a random finite time. We derive stochastic evolution equations for the mean-field compartmental populations with mortality of walkers and delayed transitions among the compartments. From linear stability analysis, we derive the basic reproduction numbers R M , R 0 with and without mortality, respectively, and prove that R M < R 0 . For R M , R 0 > 1 the healthy state is unstable whereas for zero mortality a stable endemic equilibrium exists (independent of the initial conditions) which we obtained explicitly. We observe that the solutions of the random walk simulations in the considered networks agree well with the mean-field solutions for strongly connected graph topologies, whereas less well for weakly connected structures and for diseases with high mortality.
       


### [Towards an extension of Fault Trees in the Predictive Maintenance Scenario](https://arxiv.org/abs/2403.13785)

**Authors:**
Roberta De Fazio, Stefano Marrone, Laura Verde, Vincenzo Reccia, Paolo Valletta

**Abstract:**
One of the most appreciated features of Fault Trees (FTs) is their simplicity, making them fit into industrial processes. As such processes evolve in time, considering new aspects of large modern systems, modelling techniques based on FTs have adapted to these needs. This paper proposes an extension of FTs to take into account the problem of Predictive Maintenance, one of the challenges of the modern dependability field of study. The paper sketches the Predictive Fault Tree language and proposes some use cases to support their modelling and analysis in concrete industrial settings.
       


### [IIP-Mixer:Intra-Inter Patch Mixing Architecture for Battery Remaining Useful Life Prediction](https://arxiv.org/abs/2403.18379)

**Authors:**
Guangzai Ye, Li Feng, Jianlan Guo, Yuqiang Chen

**Abstract:**
Accurately estimating the Remaining Useful Life (RUL) of lithium-ion batteries is crucial for maintaining the safe and stable operation of rechargeable battery management systems. However, this task is often challenging due to the complex temporal dynamics involved. Recently, attention-based networks, such as Transformers and Informer, have been the popular architecture in time series forecasting. Despite their effectiveness, these models with abundant parameters necessitate substantial training time to unravel temporal patterns. To tackle these challenges, we propose a simple MLP-Mixer-based architecture named 'Intra-Inter Patch Mixer' (IIP-Mixer), which is an architecture based exclusively on multi-layer perceptrons (MLPs), extracting information by mixing operations along both intra-patch and inter-patch dimensions for battery RUL prediction. The proposed IIP-Mixer comprises parallel dual-head mixer layers: the intra-patch mixing MLP, capturing local temporal patterns in the short-term period, and the inter-patch mixing MLP, capturing global temporal patterns in the long-term period. Notably, to address the varying importance of features in RUL prediction, we introduce a weighted loss function in the MLP-Mixer-based architecture, marking the first time such an approach has been employed. Our experiments demonstrate that IIP-Mixer achieves competitive performance in battery RUL prediction, outperforming other popular time-series frameworks
       


### [The State of Lithium-Ion Battery Health Prognostics in the CPS Era](https://arxiv.org/abs/2403.19816)

**Authors:**
Gaurav Shinde, Rohan Mohapatra, Pooja Krishan, Harish Garg, Srikanth Prabhu, Sanchari Das, Mohammad Masum, Saptarshi Sengupta

**Abstract:**
Lithium-ion batteries (Li-ion) have revolutionized energy storage technology, becoming integral to our daily lives by powering a diverse range of devices and applications. Their high energy density, fast power response, recyclability, and mobility advantages have made them the preferred choice for numerous sectors. This paper explores the seamless integration of Prognostics and Health Management within batteries, presenting a multidisciplinary approach that enhances the reliability, safety, and performance of these powerhouses. Remaining useful life (RUL), a critical concept in prognostics, is examined in depth, emphasizing its role in predicting component failure before it occurs. The paper reviews various RUL prediction methods, from traditional models to cutting-edge data-driven techniques. Furthermore, it highlights the paradigm shift toward deep learning architectures within the field of Li-ion battery health prognostics, elucidating the pivotal role of deep learning in addressing battery system complexities. Practical applications of PHM across industries are also explored, offering readers insights into real-world implementations.This paper serves as a comprehensive guide, catering to both researchers and practitioners in the field of Li-ion battery PHM.
       


## April
### [Cycle Life Prediction for Lithium-ion Batteries: Machine Learning and More](https://arxiv.org/abs/2404.04049)

**Authors:**
Joachim Schaeffer, Giacomo Galuppini, Jinwook Rhyu, Patrick A. Asinger, Robin Droop, Rolf Findeisen, Richard D. Braatz

**Abstract:**
Batteries are dynamic systems with complicated nonlinear aging, highly dependent on cell design, chemistry, manufacturing, and operational conditions. Prediction of battery cycle life and estimation of aging states is important to accelerate battery R&D, testing, and to further the understanding of how batteries degrade. Beyond testing, battery management systems rely on real-time models and onboard diagnostics and prognostics for safe operation. Estimating the state of health and remaining useful life of a battery is important to optimize performance and use resources optimally.
  This tutorial begins with an overview of first-principles, machine learning, and hybrid battery models. Then, a typical pipeline for the development of interpretable machine learning models is explained and showcased for cycle life prediction from laboratory testing data. We highlight the challenges of machine learning models, motivating the incorporation of physics in hybrid modeling approaches, which are needed to decipher the aging trajectory of batteries but require more data and further work on the physics of battery degradation. The tutorial closes with a discussion on generalization and further research directions.
       


### [Generalizable Temperature Nowcasting with Physics-Constrained RNNs for Predictive Maintenance of Wind Turbine Components](https://arxiv.org/abs/2404.04126)

**Authors:**
Johannes Exenberger, Matteo Di Salvo, Thomas Hirsch, Franz Wotawa, Gerald Schweiger

**Abstract:**
Machine learning plays an important role in the operation of current wind energy production systems. One central application is predictive maintenance to increase efficiency and lower electricity costs by reducing downtimes. Integrating physics-based knowledge in neural networks to enforce their physical plausibilty is a promising method to improve current approaches, but incomplete system information often impedes their application in real world scenarios. We describe a simple and efficient way for physics-constrained deep learning-based predictive maintenance for wind turbine gearbox bearings with partial system knowledge. The approach is based on temperature nowcasting constrained by physics, where unknown system coefficients are treated as learnable neural network parameters. Results show improved generalization performance to unseen environments compared to a baseline neural network, which is especially important in low data scenarios often encountered in real-world applications.
       


### [Physics-Informed Machine Learning for Battery Degradation Diagnostics: A Comparison of State-of-the-Art Methods](https://arxiv.org/abs/2404.04429)

**Authors:**
Sina Navidi, Adam Thelen, Tingkai Li, Chao Hu

**Abstract:**
Monitoring the health of lithium-ion batteries' internal components as they age is crucial for optimizing cell design and usage control strategies. However, quantifying component-level degradation typically involves aging many cells and destructively analyzing them throughout the aging test, limiting the scope of quantifiable degradation to the test conditions and duration. Fortunately, recent advances in physics-informed machine learning (PIML) for modeling and predicting the battery state of health demonstrate the feasibility of building models to predict the long-term degradation of a lithium-ion battery cell's major components using only short-term aging test data by leveraging physics. In this paper, we present four approaches for building physics-informed machine learning models and comprehensively compare them, considering accuracy, complexity, ease-of-implementation, and their ability to extrapolate to untested conditions. We delve into the details of each physics-informed machine learning method, providing insights specific to implementing them on small battery aging datasets. Our study utilizes long-term cycle aging data from 24 implantable-grade lithium-ion cells subjected to varying temperatures and C-rates over four years. This paper aims to facilitate the selection of an appropriate physics-informed machine learning method for predicting long-term degradation in lithium-ion batteries, using short-term aging data while also providing insights about when to choose which method for general predictive purposes.
       


### [Mixup Domain Adaptations for Dynamic Remaining Useful Life Predictions](https://arxiv.org/abs/2404.04824)

**Authors:**
Muhammad Tanzil Furqon, Mahardhika Pratama, Lin Liu, Habibullah, Kutluyil Dogancay

**Abstract:**
Remaining Useful Life (RUL) predictions play vital role for asset planning and maintenance leading to many benefits to industries such as reduced downtime, low maintenance costs, etc. Although various efforts have been devoted to study this topic, most existing works are restricted for i.i.d conditions assuming the same condition of the training phase and the deployment phase. This paper proposes a solution to this problem where a mix-up domain adaptation (MDAN) is put forward. MDAN encompasses a three-staged mechanism where the mix-up strategy is not only performed to regularize the source and target domains but also applied to establish an intermediate mix-up domain where the source and target domains are aligned. The self-supervised learning strategy is implemented to prevent the supervision collapse problem. Rigorous evaluations have been performed where MDAN is compared to recently published works for dynamic RUL predictions. MDAN outperforms its counterparts with substantial margins in 12 out of 12 cases. In addition, MDAN is evaluated with the bearing machine dataset where it beats prior art with significant gaps in 8 of 12 cases. Source codes of MDAN are made publicly available in \url{https://github.com/furqon3009/MDAN}.
       


### [Physics-Enhanced Graph Neural Networks For Soft Sensing in Industrial Internet of Things](https://arxiv.org/abs/2404.08061)

**Authors:**
Keivan Faghih Niresi, Hugo Bissig, Henri Baumann, Olga Fink

**Abstract:**
The Industrial Internet of Things (IIoT) is reshaping manufacturing, industrial processes, and infrastructure management. By fostering new levels of automation, efficiency, and predictive maintenance, IIoT is transforming traditional industries into intelligent, seamlessly interconnected ecosystems. However, achieving highly reliable IIoT can be hindered by factors such as the cost of installing large numbers of sensors, limitations in retrofitting existing systems with sensors, or harsh environmental conditions that may make sensor installation impractical. Soft (virtual) sensing leverages mathematical models to estimate variables from physical sensor data, offering a solution to these challenges. Data-driven and physics-based modeling are the two main methodologies widely used for soft sensing. The choice between these strategies depends on the complexity of the underlying system, with the data-driven approach often being preferred when the physics-based inference models are intricate and present challenges for state estimation. However, conventional deep learning models are typically hindered by their inability to explicitly represent the complex interactions among various sensors. To address this limitation, we adopt Graph Neural Networks (GNNs), renowned for their ability to effectively capture the complex relationships between sensor measurements. In this research, we propose physics-enhanced GNNs, which integrate principles of physics into graph-based methodologies. This is achieved by augmenting additional nodes in the input graph derived from the underlying characteristics of the physical processes. Our evaluation of the proposed methodology on the case study of district heating networks reveals significant improvements over purely data-driven GNNs, even in the presence of noise and parameter inaccuracies.
       


### [An Iterative Refinement Approach for the Rolling Stock Rotation Problem with Predictive Maintenance](https://arxiv.org/abs/2404.08367)

**Authors:**
Felix Prause, Ralf Borndörfer

**Abstract:**
The rolling stock rotation problem with predictive maintenance (RSRP-PdM) involves the assignment of trips to a fleet of vehicles with integrated maintenance scheduling based on the predicted failure probability of the vehicles. These probabilities are determined by the health states of the vehicles, which are considered to be random variables distributed by a parameterized family of probability distribution functions. During the operation of the trips, the corresponding parameters get updated. In this article, we present a dual solution approach for RSRP-PdM and generalize a linear programming based lower bound for this problem to families of probability distribution functions with more than one parameter. For this purpose, we define a rounding function that allows for a consistent underestimation of the parameters and model the problem by a state-expanded event-graph in which the possible states are restricted to a discrete set. This induces a flow problem that is solved by an integer linear program. We show that the iterative refinement of the underlying discretization leads to solutions that converge from below to an optimal solution of the original instance. Thus, the linear relaxation of the considered integer linear program results in a lower bound for RSRP-PdM. Finally, we report on the results of computational experiments conducted on a library of test instances.
       


### [CARE to Compare: A real-world dataset for anomaly detection in wind turbine data](https://arxiv.org/abs/2404.10320)

**Authors:**
Christian Gück, Cyriana M. A. Roelofs, Stefan Faulstich

**Abstract:**
Anomaly detection plays a crucial role in the field of predictive maintenance for wind turbines, yet the comparison of different algorithms poses a difficult task because domain specific public datasets are scarce. Many comparisons of different approaches either use benchmarks composed of data from many different domains, inaccessible data or one of the few publicly available datasets which lack detailed information about the faults. Moreover, many publications highlight a couple of case studies where fault detection was successful. With this paper we publish a high quality dataset that contains data from 36 wind turbines across 3 different wind farms as well as the most detailed fault information of any public wind turbine dataset as far as we know. The new dataset contains 89 years worth of real-world operating data of wind turbines, distributed across 44 labeled time frames for anomalies that led up to faults, as well as 51 time series representing normal behavior. Additionally, the quality of training data is ensured by turbine-status-based labels for each data point. Furthermore, we propose a new scoring method, called CARE (Coverage, Accuracy, Reliability and Earliness), which takes advantage of the information depth that is present in the dataset to identify a good all-around anomaly detection model. This score considers the anomaly detection performance, the ability to recognize normal behavior properly and the capability to raise as few false alarms as possible while simultaneously detecting anomalies early.
       


### [Analytical results for uncertainty propagation through trained machine learning regression models](https://arxiv.org/abs/2404.11224)

**Author:**
Andrew Thompson

**Abstract:**
Machine learning (ML) models are increasingly being used in metrology applications. However, for ML models to be credible in a metrology context they should be accompanied by principled uncertainty quantification. This paper addresses the challenge of uncertainty propagation through trained/fixed machine learning (ML) regression models. Analytical expressions for the mean and variance of the model output are obtained/presented for certain input data distributions and for a variety of ML models. Our results cover several popular ML models including linear regression, penalised linear regression, kernel ridge regression, Gaussian Processes (GPs), support vector machines (SVMs) and relevance vector machines (RVMs). We present numerical experiments in which we validate our methods and compare them with a Monte Carlo approach from a computational efficiency point of view. We also illustrate our methods in the context of a metrology application, namely modelling the state-of-health of lithium-ion cells based upon Electrical Impedance Spectroscopy (EIS) data
       


### [A Data-Driven Condition Monitoring Method for Capacitor in Modular Multilevel Converter (MMC)](https://arxiv.org/abs/2404.13399)

**Authors:**
Shuyu Ou, Mahyar Hassanifar, Martin Votava, Marius Langwasser, Marco Liserre, Ariya Sangwongwanich, Subham Sahoo, Frede Blaabjerg

**Abstract:**
The modular multilevel converter (MMC) is a topology that consists of a high number of capacitors, and degradation of capacitors can lead to converter malfunction, limiting the overall system lifetime. Condition monitoring methods can be applied to assess the health status of capacitors and realize predictive maintenance to improve reliability. Current research works for condition monitoring of capacitors in an MMC mainly monitor either capacitance or equivalent series resistance (ESR), while these two health indicators can shift at different speeds and lead to different end-of-life times. Hence, monitoring only one of these parameters may lead to unreliable health status evaluation. This paper proposes a data-driven method to estimate capacitance and ESR at the same time, in which particle swarm optimization (PSO) is leveraged to update the obtained estimations. Then, the results of the estimations are used to predict the sub-module voltage, which is based on a capacitor voltage equation. Furthermore, minimizing the mean square error between the predicted and actual measured voltage makes the estimations closer to the actual values. The effectiveness and feasibility of the proposed method are validated through simulations and experiments.
       


### [Revolutionizing System Reliability: The Role of AI in Predictive Maintenance Strategies](https://arxiv.org/abs/2404.13454)

**Authors:**
Michael Bidollahkhani, Julian M. Kunkel

**Abstract:**
The landscape of maintenance in distributed systems is rapidly evolving with the integration of Artificial Intelligence (AI). Also, as the complexity of computing continuum systems intensifies, the role of AI in predictive maintenance (Pd.M.) becomes increasingly pivotal. This paper presents a comprehensive survey of the current state of Pd.M. in the computing continuum, with a focus on the combination of scalable AI technologies. Recognizing the limitations of traditional maintenance practices in the face of increasingly complex and heterogenous computing continuum systems, the study explores how AI, especially machine learning and neural networks, is being used to enhance Pd.M. strategies. The survey encompasses a thorough review of existing literature, highlighting key advancements, methodologies, and case studies in the field. It critically examines the role of AI in improving prediction accuracy for system failures and in optimizing maintenance schedules, thereby contributing to reduced downtime and enhanced system longevity. By synthesizing findings from the latest advancements in the field, the article provides insights into the effectiveness and challenges of implementing AI-driven predictive maintenance. It underscores the evolution of maintenance practices in response to technological advancements and the growing complexity of computing continuum systems. The conclusions drawn from this survey are instrumental for practitioners and researchers in understanding the current landscape and future directions of Pd.M. in distributed systems. It emphasizes the need for continued research and development in this area, pointing towards a trend of more intelligent, efficient, and cost-effective maintenance solutions in the era of AI.
       


### [A Neuro-Symbolic Explainer for Rare Events: A Case Study on Predictive Maintenance](https://arxiv.org/abs/2404.14455)

**Authors:**
João Gama, Rita P. Ribeiro, Saulo Mastelini, Narjes Davarid, Bruno Veloso

**Abstract:**
Predictive Maintenance applications are increasingly complex, with interactions between many components. Black box models are popular approaches based on deep learning techniques due to their predictive accuracy. This paper proposes a neural-symbolic architecture that uses an online rule-learning algorithm to explain when the black box model predicts failures. The proposed system solves two problems in parallel: anomaly detection and explanation of the anomaly. For the first problem, we use an unsupervised state of the art autoencoder. For the second problem, we train a rule learning system that learns a mapping from the input features to the autoencoder reconstruction error. Both systems run online and in parallel. The autoencoder signals an alarm for the examples with a reconstruction error that exceeds a threshold. The causes of the signal alarm are hard for humans to understand because they result from a non linear combination of sensor data. The rule that triggers that example describes the relationship between the input features and the autoencoder reconstruction error. The rule explains the failure signal by indicating which sensors contribute to the alarm and allowing the identification of the component involved in the failure. The system can present global explanations for the black box model and local explanations for why the black box model predicts a failure. We evaluate the proposed system in a real-world case study of Metro do Porto and provide explanations that illustrate its benefits.
       


### [Predictive Intent Maintenance with Intent Drift Detection in Next Generation Network](https://arxiv.org/abs/2404.15091)

**Authors:**
Chukwuemeka Muonagor, Mounir Bensalem, Admela Jukan

**Abstract:**
Intent-Based Networking (IBN) is a known concept for enabling the autonomous configuration and self-adaptation of networks. One of the major issues in IBN is maintaining the applied intent due the effects of drifts over time, which is the gradual degradation in the fulfillment of the intents, before they fail. Despite its critical role to intent assurance and maintenance, intent drift detection was largely overlooked in the literature. To fill this gap, we propose an intent drift detection algorithm for predictive maintenance of intents which can use various unsupervised learning techniques (Affinity Propagation, DBSCAN, Gaussian Mixture Models, Hierarchical clustering, K-Means clustering, OPTICS, One-Class SVM), here applied and comparatively analyzed due to their simplicity, yet efficiency in detecting drifts. The results show that DBSCAN is the best model for detecting the intent drifts. The worst performance is exhibited by the Affinity Propagation model, reflected in its poorest accuracy and latency values.
       


### [Cycling into the workshop: predictive maintenance for Barcelona's bike-sharing system](https://arxiv.org/abs/2404.17217)

**Authors:**
Jordi Grau-Escolano, Aleix Bassolas, Julian Vicens

**Abstract:**
Bike-sharing systems have emerged as a significant element of urban mobility, providing an environmentally friendly transportation alternative. With the increasing integration of electric bikes alongside mechanical bikes, it is crucial to illuminate distinct usage patterns and their impact on maintenance. Accordingly, this research aims to develop a comprehensive understanding of mobility dynamics, distinguishing between different mobility modes, and introducing a novel predictive maintenance system tailored for bikes. By utilising a combination of trip information and maintenance data from Barcelona's bike-sharing system, Bicing, this study conducts an extensive analysis of mobility patterns and their relationship to failures of bike components. To accurately predict maintenance needs for essential bike parts, this research delves into various mobility metrics and applies statistical and machine learning survival models, including deep learning models. Due to their complexity, and with the objective of bolstering confidence in the system's predictions, interpretability techniques explain the main predictors of maintenance needs. The analysis reveals marked differences in the usage patterns of mechanical bikes and electric bikes, with a growing user preference for the latter despite their extra costs. These differences in mobility were found to have a considerable impact on the maintenance needs within the bike-sharing system. Moreover, the predictive maintenance models proved effective in forecasting these maintenance needs, capable of operating across an entire bike fleet. Despite challenges such as approximated bike usage metrics and data imbalances, the study successfully showcases the feasibility of an accurate predictive maintenance system capable of improving operational costs, bike availability, and security.
       


## May
### [AI for Manufacturing and Healthcare: a chemistry and engineering perspective](https://arxiv.org/abs/2405.01520)

**Authors:**
Jihua Chen, Yue Yuan, Amir Koushyar Ziabari, Xuan Xu, Honghai Zhang, Panagiotis Christakopoulos, Peter V. Bonnesen, Ilia N. Ivanov, Panchapakesan Ganesh, Chen Wang, Karen Patino Jaimes, Guang Yang, Rajeev Kumar, Bobby G. Sumpter, Rigoberto Advincula

**Abstract:**
Artificial Intelligence (AI) approaches are increasingly being applied to more and more domains of Science, Engineering, Chemistry, and Industries to not only improve efficiencies and enhance productivity, but also enable new capabilities. The new opportunities range from automated molecule design and screening, properties prediction, gaining insights of chemical reactions, to computer-aided design, predictive maintenance of systems, robotics, and autonomous vehicles. This review focuses on the new applications of AI in manufacturing and healthcare. For the Manufacturing Industries, we focus on AI and algorithms for (1) Battery, (2) Flow Chemistry, (3) Additive Manufacturing, (4) Sensors, and (5) Machine Vision. For Healthcare applications, we focus on: (1) Medical Vision (2) Diagnosis, (3) Protein Design, and (4) Drug Discovery. In the end, related topics are discussed, including physics integrated machine learning, model explainability, security, and governance during model deployment.
       


### [A probabilistic estimation of remaining useful life from censored time-to-event data](https://arxiv.org/abs/2405.01614)

**Authors:**
Christian Marius Lillelund, Fernando Pannullo, Morten Opprud Jakobsen, Manuel Morante, Christian Fischer Pedersen

**Abstract:**
Predicting the remaining useful life (RUL) of ball bearings plays an important role in predictive maintenance. A common definition of the RUL is the time until a bearing is no longer functional, which we denote as an event, and many data-driven methods have been proposed to predict the RUL. However, few studies have addressed the problem of censored data, where this event of interest is not observed, and simply ignoring these observations can lead to an overestimation of the failure risk. In this paper, we propose a probabilistic estimation of RUL using survival analysis that supports censored data. First, we analyze sensor readings from ball bearings in the frequency domain and annotate when a bearing starts to deteriorate by calculating the Kullback-Leibler (KL) divergence between the probability density function (PDF) of the current process and a reference PDF. Second, we train several survival models on the annotated bearing dataset, capable of predicting the RUL over a finite time horizon using the survival function. This function is guaranteed to be strictly monotonically decreasing and is an intuitive estimation of the remaining lifetime. We demonstrate our approach in the XJTU-SY dataset using cross-validation and find that Random Survival Forests consistently outperforms both non-neural networks and neural networks in terms of the mean absolute error (MAE). Our work encourages the inclusion of censored data in predictive maintenance models and highlights the unique advantages that survival analysis offers when it comes to probabilistic RUL estimation and early fault detection.
       


### [Temporal and Heterogeneous Graph Neural Network for Remaining Useful Life Prediction](https://arxiv.org/abs/2405.04336)

**Authors:**
Zhihao Wen, Yuan Fang, Pengcheng Wei, Fayao Liu, Zhenghua Chen, Min Wu

**Abstract:**
Predicting Remaining Useful Life (RUL) plays a crucial role in the prognostics and health management of industrial systems that involve a variety of interrelated sensors. Given a constant stream of time series sensory data from such systems, deep learning models have risen to prominence at identifying complex, nonlinear temporal dependencies in these data. In addition to the temporal dependencies of individual sensors, spatial dependencies emerge as important correlations among these sensors, which can be naturally modelled by a temporal graph that describes time-varying spatial relationships. However, the majority of existing studies have relied on capturing discrete snapshots of this temporal graph, a coarse-grained approach that leads to loss of temporal information. Moreover, given the variety of heterogeneous sensors, it becomes vital that such inherent heterogeneity is leveraged for RUL prediction in temporal sensor graphs. To capture the nuances of the temporal and spatial relationships and heterogeneous characteristics in an interconnected graph of sensors, we introduce a novel model named Temporal and Heterogeneous Graph Neural Networks (THGNN). Specifically, THGNN aggregates historical data from neighboring nodes to accurately capture the temporal dynamics and spatial correlations within the stream of sensor data in a fine-grained manner. Moreover, the model leverages Feature-wise Linear Modulation (FiLM) to address the diversity of sensor types, significantly improving the model's capacity to learn the heterogeneity in the data sources. Finally, we have validated the effectiveness of our approach through comprehensive experiments. Our empirical findings demonstrate significant advancements on the N-CMAPSS dataset, achieving improvements of up to 19.2% and 31.6% in terms of two different evaluation metrics over state-of-the-art methods.
       


### [Health Index Estimation Through Integration of General Knowledge with Unsupervised Learning](https://arxiv.org/abs/2405.04990)

**Authors:**
Kristupas Bajarunas, Marcia L. Baptista, Kai Goebel, Manuel A. Chao

**Abstract:**
Accurately estimating a Health Index (HI) from condition monitoring data (CM) is essential for reliable and interpretable prognostics and health management (PHM) in complex systems. In most scenarios, complex systems operate under varying operating conditions and can exhibit different fault modes, making unsupervised inference of an HI from CM data a significant challenge. Hybrid models combining prior knowledge about degradation with deep learning models have been proposed to overcome this challenge. However, previously suggested hybrid models for HI estimation usually rely heavily on system-specific information, limiting their transferability to other systems. In this work, we propose an unsupervised hybrid method for HI estimation that integrates general knowledge about degradation into the convolutional autoencoder's model architecture and learning algorithm, enhancing its applicability across various systems. The effectiveness of the proposed method is demonstrated in two case studies from different domains: turbofan engines and lithium batteries. The results show that the proposed method outperforms other competitive alternatives, including residual-based methods, in terms of HI quality and their utility for Remaining Useful Life (RUL) predictions. The case studies also highlight the comparable performance of our proposed method with a supervised model trained with HI labels.
       


### [Dynamic classifier auditing by unsupervised anomaly detection methods: an application in packaging industry predictive maintenance](https://arxiv.org/abs/2405.11960)

**Authors:**
Fernando Mateo, Joan Vila-Francés, Emilio Soria-Olivas, Marcelino Martínez-Sober Juan Gómez-Sanchis, Antonio-José Serrano-López

**Abstract:**
Predictive maintenance in manufacturing industry applications is a challenging research field. Packaging machines are widely used in a large number of logistic companies' warehouses and must be working uninterruptedly. Traditionally, preventive maintenance strategies have been carried out to improve the performance of these machines. However, this kind of policies does not take into account the information provided by the sensors implemented in the machines. This paper presents an expert system for the automatic estimation of work orders to implement predictive maintenance policies for packaging machines. The key idea is that, from a set of alarms related to sensors implemented in the machine, the expert system should take a maintenance action while optimizing the response time. The work order estimator will act as a classifier, yielding a binary decision of whether a machine must undergo a maintenance action by a technician or not, followed by an unsupervised anomaly detection-based filtering stage to audit the classifier's output. The methods used for anomaly detection were: One-Class Support Vector Machine (OCSVM), Minimum Covariance Determinant (MCD) and a majority (hard) voting ensemble of them. All anomaly detection methods improve the performance of the baseline classifer but the best performance in terms of F1 score was obtained by the majority voting ensemble.
       


### [The Case for DeepSOH: Addressing Path Dependency for Remaining Useful Life](https://arxiv.org/abs/2405.12028)

**Authors:**
Hamidreza Movahedi, Andrew Weng, Sravan Pannala, Jason B. Siegel, Anna G. Stefanopoulou

**Abstract:**
The battery state of health (SOH) based on capacity fade and resistance increase is not sufficient for predicting Remaining Useful life (RUL). The electrochemical community blames the path-dependency of the battery degradation mechanisms for our inability to forecast the degradation. The control community knows that the path-dependency is addressed by full state estimation. We show that even the electrode-specific SOH (eSOH) estimation is not enough to fully define the degradation states by simulating infinite possible degradation trajectories and remaining useful lives (RUL) from a unique eSOH. We finally define the deepSOH states that capture the individual contributions of all the common degradation mechanisms, namely, SEI, plating, and mechanical fracture to the loss of lithium inventory. We show that the addition of cell expansion measurement may allow us to estimate the deepSOH and predict the remaining useful life.
       


### [Spatio-temporal Attention-based Hidden Physics-informed Neural Network for Remaining Useful Life Prediction](https://arxiv.org/abs/2405.12377)

**Authors:**
Feilong Jiang, Xiaonan Hou, Min Xia

**Abstract:**
Predicting the Remaining Useful Life (RUL) is essential in Prognostic Health Management (PHM) for industrial systems. Although deep learning approaches have achieved considerable success in predicting RUL, challenges such as low prediction accuracy and interpretability pose significant challenges, hindering their practical implementation. In this work, we introduce a Spatio-temporal Attention-based Hidden Physics-informed Neural Network (STA-HPINN) for RUL prediction, which can utilize the associated physics of the system degradation. The spatio-temporal attention mechanism can extract important features from the input data. With the self-attention mechanism on both the sensor dimension and time step dimension, the proposed model can effectively extract degradation information. The hidden physics-informed neural network is utilized to capture the physics mechanisms that govern the evolution of RUL. With the constraint of physics, the model can achieve higher accuracy and reasonable predictions. The approach is validated on a benchmark dataset, demonstrating exceptional performance when compared to cutting-edge methods, especially in the case of complex conditions.
       


### [Artificial Intelligence Approaches for Predictive Maintenance in the Steel Industry: A Survey](https://arxiv.org/abs/2405.12785)

**Authors:**
Jakub Jakubowski, Natalia Wojak-Strzelecka, Rita P. Ribeiro, Sepideh Pashami, Szymon Bobek, Joao Gama, Grzegorz J Nalepa

**Abstract:**
Predictive Maintenance (PdM) emerged as one of the pillars of Industry 4.0, and became crucial for enhancing operational efficiency, allowing to minimize downtime, extend lifespan of equipment, and prevent failures. A wide range of PdM tasks can be performed using Artificial Intelligence (AI) methods, which often use data generated from industrial sensors. The steel industry, which is an important branch of the global economy, is one of the potential beneficiaries of this trend, given its large environmental footprint, the globalized nature of the market, and the demanding working conditions. This survey synthesizes the current state of knowledge in the field of AI-based PdM within the steel industry and is addressed to researchers and practitioners. We identified 219 articles related to this topic and formulated five research questions, allowing us to gain a global perspective on current trends and the main research gaps. We examined equipment and facilities subjected to PdM, determined common PdM approaches, and identified trends in the AI methods used to develop these solutions. We explored the characteristics of the data used in the surveyed articles and assessed the practical implications of the research presented there. Most of the research focuses on the blast furnace or hot rolling, using data from industrial sensors. Current trends show increasing interest in the domain, especially in the use of deep learning. The main challenges include implementing the proposed methods in a production environment, incorporating them into maintenance plans, and enhancing the accessibility and reproducibility of the research.
       


### [Towards a Probabilistic Fusion Approach for Robust Battery Prognostics](https://arxiv.org/abs/2405.15292)

**Authors:**
Jokin Alcibar, Jose I. Aizpurua, Ekhi Zugasti

**Abstract:**
Batteries are a key enabling technology for the decarbonization of transport and energy sectors. The safe and reliable operation of batteries is crucial for battery-powered systems. In this direction, the development of accurate and robust battery state-of-health prognostics models can unlock the potential of autonomous systems for complex, remote and reliable operations. The combination of Neural Networks, Bayesian modelling concepts and ensemble learning strategies, form a valuable prognostics framework to combine uncertainty in a robust and accurate manner. Accordingly, this paper introduces a Bayesian ensemble learning approach to predict the capacity depletion of lithium-ion batteries. The approach accurately predicts the capacity fade and quantifies the uncertainty associated with battery design and degradation processes. The proposed Bayesian ensemble methodology employs a stacking technique, integrating multiple Bayesian neural networks (BNNs) as base learners, which have been trained on data diversity. The proposed method has been validated using a battery aging dataset collected by the NASA Ames Prognostics Center of Excellence. Obtained results demonstrate the improved accuracy and robustness of the proposed probabilistic fusion approach with respect to (i) a single BNN model and (ii) a classical stacking strategy based on different BNNs.
       


### [DETECTA 2.0: Research into non-intrusive methodologies supported by Industry 4.0 enabling technologies for predictive and cyber-secure maintenance in SMEs](https://arxiv.org/abs/2405.15832)

**Authors:**
Álvaro Huertas-García, Javier Muñoz, Enrique De Miguel Ambite, Marcos Avilés Camarmas, José Félix Ovejero

**Abstract:**
The integration of predictive maintenance and cybersecurity represents a transformative advancement for small and medium-sized enterprises (SMEs) operating within the Industry 4.0 paradigm. Despite their economic importance, SMEs often face significant challenges in adopting advanced technologies due to resource constraints and knowledge gaps. The DETECTA 2.0 project addresses these hurdles by developing an innovative system that harmonizes real-time anomaly detection, sophisticated analytics, and predictive forecasting capabilities.
  The system employs a semi-supervised methodology, combining unsupervised anomaly detection with supervised learning techniques. This approach enables more agile and cost-effective development of AI detection systems, significantly reducing the time required for manual case review.
  At the core lies a Digital Twin interface, providing intuitive real-time visualizations of machine states and detected anomalies. Leveraging cutting-edge AI engines, the system intelligently categorizes anomalies based on observed patterns, differentiating between technical errors and potential cybersecurity incidents. This discernment is fortified by detailed analytics, including certainty levels that enhance alert reliability and minimize false positives.
  The predictive engine uses advanced time series algorithms like N-HiTS to forecast future machine utilization trends. This proactive approach optimizes maintenance planning, enhances cybersecurity measures, and minimizes unplanned downtimes despite variable production processes.
  With its modular architecture enabling seamless integration across industrial setups and low implementation costs, DETECTA 2.0 presents an attractive solution for SMEs to strengthen their predictive maintenance and cybersecurity strategies.
       


### [Pattern-Based Time-Series Risk Scoring for Anomaly Detection and Alert Filtering -- A Predictive Maintenance Case Study](https://arxiv.org/abs/2405.17488)

**Author:**
Elad Liebman

**Abstract:**
Fault detection is a key challenge in the management of complex systems. In the context of SparkCognition's efforts towards predictive maintenance in large scale industrial systems, this problem is often framed in terms of anomaly detection - identifying patterns of behavior in the data which deviate from normal. Patterns of normal behavior aren't captured simply in the coarse statistics of measured signals. Rather, the multivariate sequential pattern itself can be indicative of normal vs. abnormal behavior. For this reason, normal behavior modeling that relies on snapshots of the data without taking into account temporal relationships as they evolve would be lacking. However, common strategies for dealing with temporal dependence, such as Recurrent Neural Networks or attention mechanisms are oftentimes computationally expensive and difficult to train. In this paper, we propose a fast and efficient approach to anomaly detection and alert filtering based on sequential pattern similarities. In our empirical analysis section, we show how this approach can be leveraged for a variety of purposes involving anomaly detection on a large scale real-world industrial system. Subsequently, we test our approach on a publicly-available dataset in order to establish its general applicability and robustness compared to a state-of-the-art baseline. We also demonstrate an efficient way of optimizing the framework based on an alert recall objective function.
       


### [Interpretable Prognostics with Concept Bottleneck Models](https://arxiv.org/abs/2405.17575)

**Authors:**
Florent Forest, Katharina Rombach, Olga Fink

**Abstract:**
Deep learning approaches have recently been extensively explored for the prognostics of industrial assets. However, they still suffer from a lack of interpretability, which hinders their adoption in safety-critical applications. To improve their trustworthiness, explainable AI (XAI) techniques have been applied in prognostics, primarily to quantify the importance of input variables for predicting the remaining useful life (RUL) using post-hoc attribution methods. In this work, we propose the application of Concept Bottleneck Models (CBMs), a family of inherently interpretable neural network architectures based on concept explanations, to the task of RUL prediction. Unlike attribution methods, which explain decisions in terms of low-level input features, concepts represent high-level information that is easily understandable by users. Moreover, once verified in actual applications, CBMs enable domain experts to intervene on the concept activations at test-time. We propose using the different degradation modes of an asset as intermediate concepts. Our case studies on the New Commercial Modular AeroPropulsion System Simulation (N-CMAPSS) aircraft engine dataset for RUL prediction demonstrate that the performance of CBMs can be on par or superior to black-box models, while being more interpretable, even when the available labeled concepts are limited. Code available at \href{https://github.com/EPFL-IMOS/concept-prognostics/}{\url{github.com/EPFL-IMOS/concept-prognostics/}}.
       


### [Artificial Intelligence in Industry 4.0: A Review of Integration Challenges for Industrial Systems](https://arxiv.org/abs/2405.18580)

**Authors:**
Alexander Windmann, Philipp Wittenberg, Marvin Schieseck, Oliver Niggemann

**Abstract:**
In Industry 4.0, Cyber-Physical Systems (CPS) generate vast data sets that can be leveraged by Artificial Intelligence (AI) for applications including predictive maintenance and production planning. However, despite the demonstrated potential of AI, its widespread adoption in sectors like manufacturing remains limited. Our comprehensive review of recent literature, including standards and reports, pinpoints key challenges: system integration, data-related issues, managing workforce-related concerns and ensuring trustworthy AI. A quantitative analysis highlights particular challenges and topics that are important for practitioners but still need to be sufficiently investigated by academics. The paper briefly discusses existing solutions to these challenges and proposes avenues for future research. We hope that this survey serves as a resource for practitioners evaluating the cost-benefit implications of AI in CPS and for researchers aiming to address these urgent challenges.
       


### [Share Your Secrets for Privacy! Confidential Forecasting with Vertical Federated Learning](https://arxiv.org/abs/2405.20761)

**Authors:**
Aditya Shankar, Lydia Y. Chen, Jérémie Decouchant, Dimitra Gkorou, Rihan Hai

**Abstract:**
Vertical federated learning (VFL) is a promising area for time series forecasting in industrial applications, such as predictive maintenance and machine control. Critical challenges to address in manufacturing include data privacy and over-fitting on small and noisy datasets during both training and inference. Additionally, to increase industry adaptability, such forecasting models must scale well with the number of parties while ensuring strong convergence and low-tuning complexity. We address those challenges and propose 'Secret-shared Time Series Forecasting with VFL' (STV), a novel framework that exhibits the following key features: i) a privacy-preserving algorithm for forecasting with SARIMAX and autoregressive trees on vertically partitioned data; ii) serverless forecasting using secret sharing and multi-party computation; iii) novel N-party algorithms for matrix multiplication and inverse operations for direct parameter optimization, giving strong convergence with minimal hyperparameter tuning complexity. We conduct evaluations on six representative datasets from public and industry-specific contexts. Our results demonstrate that STV's forecasting accuracy is comparable to those of centralized approaches. They also show that our direct optimization can outperform centralized methods, which include state-of-the-art diffusion models and long-short-term memory, by 23.81% on forecasting accuracy. We also conduct a scalability analysis by examining the communication costs of direct and iterative optimization to navigate the choice between the two. Code and appendix are available: https://github.com/adis98/STV
       


## June
### [ContextFlow++: Generalist-Specialist Flow-based Generative Models with Mixed-Variable Context Encoding](https://arxiv.org/abs/2406.00578)

**Authors:**
Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata

**Abstract:**
Normalizing flow-based generative models have been widely used in applications where the exact density estimation is of major importance. Recent research proposes numerous methods to improve their expressivity. However, conditioning on a context is largely overlooked area in the bijective flow research. Conventional conditioning with the vector concatenation is limited to only a few flow types. More importantly, this approach cannot support a practical setup where a set of context-conditioned (specialist) models are trained with the fixed pretrained general-knowledge (generalist) model. We propose ContextFlow++ approach to overcome these limitations using an additive conditioning with explicit generalist-specialist knowledge decoupling. Furthermore, we support discrete contexts by the proposed mixed-variable architecture with context encoders. Particularly, our context encoder for discrete variables is a surjective flow from which the context-conditioned continuous variables are sampled. Our experiments on rotated MNIST-R, corrupted CIFAR-10C, real-world ATM predictive maintenance and SMAP unsupervised anomaly detection benchmarks show that the proposed ContextFlow++ offers faster stable training and achieves higher performance metrics. Our code is publicly available at https://github.com/gudovskiy/contextflow.
       


### [Aging modeling and lifetime prediction of a proton exchange membrane fuel cell using an extended Kalman filter](https://arxiv.org/abs/2406.01259)

**Authors:**
Serigne Daouda Pene, Antoine Picot, Fabrice Gamboa, Nicolas Savy, Christophe Turpin, Amine Jaafar

**Abstract:**
This article presents a methodology that aims to model and to provide predictive capabilities for the lifetime of Proton Exchange Membrane Fuel Cell (PEMFC). The approach integrates parametric identification, dynamic modeling, and Extended Kalman Filtering (EKF). The foundation is laid with the creation of a representative aging database, emphasizing specific operating conditions. Electrochemical behavior is characterized through the identification of critical parameters. The methodology extends to capture the temporal evolution of the identified parameters. We also address challenges posed by the limiting current density through a differential analysis-based modeling technique and the detection of breakpoints. This approach, involving Monte Carlo simulations, is coupled with an EKF for predicting voltage degradation. The Remaining Useful Life (RUL) is also estimated. The results show that our approach accurately predicts future voltage and RUL with very low relative errors.
       


### [Diagnostic Digital Twin for Anomaly Detection in Floating Offshore Wind Energy](https://arxiv.org/abs/2406.02775)

**Authors:**
Florian Stadtmann, Adil Rasheed

**Abstract:**
The demand for condition-based and predictive maintenance is rising across industries, especially for remote, high-value, and high-risk assets. In this article, the diagnostic digital twin concept is introduced, discussed, and implemented for a floating offshore turbine. A diagnostic digital twin is a virtual representation of an asset that combines real-time data and models to monitor damage, detect anomalies, and diagnose failures, thereby enabling condition-based and predictive maintenance. By applying diagnostic digital twins to offshore assets, unexpected failures can be alleviated, but the implementation can prove challenging. Here, a diagnostic digital twin is implemented for an operational floating offshore wind turbine. The asset is monitored through measurements. Unsupervised learning methods are employed to build a normal operation model, detect anomalies, and provide a fault diagnosis. Warnings and diagnoses are sent through text messages, and a more detailed diagnosis can be accessed in a virtual reality interface. The diagnostic digital twin successfully detected an anomaly with high confidence hours before a failure occurred. The paper concludes by discussing diagnostic digital twins in the broader context of offshore engineering. The presented approach can be generalized to other offshore assets to improve maintenance and increase the lifetime, efficiency, and sustainability of offshore assets.
       


### [Computationally Efficient Machine-Learning-Based Online Battery State of Health Estimation](https://arxiv.org/abs/2406.06151)

**Authors:**
Abhijit Kulkarni, Remus Teodorescu

**Abstract:**
A key function of battery management systems (BMS) in e-mobility applications is estimating the battery state of health (SoH) with high accuracy. This is typically achieved in commercial BMS using model-based methods. There has been considerable research in developing data-driven methods for improving the accuracy of SoH estimation. The data-driven methods are diverse and use different machine-learning (ML) or artificial intelligence (AI) based techniques. Complex AI/ML techniques are difficult to implement in low-cost microcontrollers used in BMS due to the extensive use of non-linear functions and large matrix operations. This paper proposes a computationally efficient and data-lightweight SoH estimation technique. Online impedance at four discrete frequencies is evaluated to derive the features of a linear regression problem. The proposed solution avoids complex mathematical operations and it is well-suited for online implementation in a commercial BMS. The accuracy of this method is validated on two experimental datasets and is shown to have a mean absolute error (MAE) of less than 2% across diverse training and testing data.
       


### [Revealing Predictive Maintenance Strategies from Comprehensive Data Analysis of ASTRI-Horn Historical Monitoring Data](https://arxiv.org/abs/2406.07308)

**Authors:**
Federico Incardona, Alessandro Costa, Giuseppe Leto, Kevin Munari, Giovanni Pareschi, Salvatore Scuderi, Gino Tosti

**Abstract:**
Modern telescope facilities generate data from various sources, including sensors, weather stations, LiDARs, and FRAMs. Sophisticated software architectures using the Internet of Things (IoT) and big data technologies are required to manage this data. This study explores the potential of sensor data for innovative maintenance techniques, such as predictive maintenance (PdM), to prevent downtime that can affect research. We analyzed historical data from the ASTRI-Horn Cherenkov telescope, spanning seven years, examining data patterns and variable correlations. The findings offer insights for triggering predictive maintenance model development in telescope facilities.
       


### [Tool Wear Prediction in CNC Turning Operations using Ultrasonic Microphone Arrays and CNNs](https://arxiv.org/abs/2406.08957)

**Authors:**
Jan Steckel, Arne Aerts, Erik Verreycken, Dennis Laurijssen, Walter Daems

**Abstract:**
This paper introduces a novel method for predicting tool wear in CNC turning operations, combining ultrasonic microphone arrays and convolutional neural networks (CNNs). High-frequency acoustic emissions between 0 kHz and 60 kHz are enhanced using beamforming techniques to improve the signal- to-noise ratio. The processed acoustic data is then analyzed by a CNN, which predicts the Remaining Useful Life (RUL) of cutting tools. Trained on data from 350 workpieces machined with a single carbide insert, the model can accurately predict the RUL of the carbide insert. Our results demonstrate the potential gained by integrating advanced ultrasonic sensors with deep learning for accurate predictive maintenance tasks in CNC machining.
       


### [Online-Adaptive Anomaly Detection for Defect Identification in Aircraft Assembly](https://arxiv.org/abs/2406.12698)

**Authors:**
Siddhant Shete, Dennis Mronga, Ankita Jadhav, Frank Kirchner

**Abstract:**
Anomaly detection deals with detecting deviations from established patterns within data. It has various applications like autonomous driving, predictive maintenance, and medical diagnosis. To improve anomaly detection accuracy, transfer learning can be applied to large, pre-trained models and adapt them to the specific application context. In this paper, we propose a novel framework for online-adaptive anomaly detection using transfer learning. The approach adapts to different environments by selecting visually similar training images and online fitting a normality model to EfficientNet features extracted from the training subset. Anomaly detection is then performed by computing the Mahalanobis distance between the normality model and the test image features. Different similarity measures (SIFT/FLANN, Cosine) and normality models (MVG, OCSVM) are employed and compared with each other. We evaluate the approach on different anomaly detection benchmarks and data collected in controlled laboratory settings. Experimental results showcase a detection accuracy exceeding 0.975, outperforming the state-of-the-art ET-NET approach.
       


### [The Significance of Latent Data Divergence in Predicting System Degradation](https://arxiv.org/abs/2406.12914)

**Authors:**
Miguel Fernandes, Catarina Silva, Alberto Cardoso, Bernardete Ribeiro

**Abstract:**
Condition-Based Maintenance is pivotal in enabling the early detection of potential failures in engineering systems, where precise prediction of the Remaining Useful Life is essential for effective maintenance and operation. However, a predominant focus in the field centers on predicting the Remaining Useful Life using unprocessed or minimally processed data, frequently neglecting the intricate dynamics inherent in the dataset. In this work we introduce a novel methodology grounded in the analysis of statistical similarity within latent data from system components. Leveraging a specifically designed architecture based on a Vector Quantized Variational Autoencoder, we create a sequence of discrete vectors which is used to estimate system-specific priors. We infer the similarity between systems by evaluating the divergence of these priors, offering a nuanced understanding of individual system behaviors. The efficacy of our approach is demonstrated through experiments on the NASA commercial modular aero-propulsion system simulation (C-MAPSS) dataset. Our validation not only underscores the potential of our method in advancing the study of latent statistical divergence but also demonstrates its superiority over existing techniques.
       


### [State-of-the-Art Review: The Use of Digital Twins to Support Artificial Intelligence-Guided Predictive Maintenance](https://arxiv.org/abs/2406.13117)

**Authors:**
Sizhe Ma, Katherine A. Flanigan, Mario Bergés

**Abstract:**
In recent years, predictive maintenance (PMx) has gained prominence for its potential to enhance efficiency, automation, accuracy, and cost-effectiveness while reducing human involvement. Importantly, PMx has evolved in tandem with digital advancements, such as Big Data and the Internet of Things (IOT). These technological strides have enabled Artificial Intelligence (AI) to revolutionize PMx processes, with increasing capacities for real-time automation of monitoring, analysis, and prediction tasks. However, PMx still faces challenges such as poor explainability and sample inefficiency in data-driven methods and high complexity in physics-based models, hindering broader adoption. This paper posits that Digital Twins (DTs) can be integrated into PMx to overcome these challenges, paving the way for more automated PMx applications across various stakeholders. Despite their potential, current DTs have not fully matured to bridge existing gaps. Our paper provides a comprehensive roadmap for DT evolution, addressing current limitations to foster large-scale automated PMx progression. We structure our approach in three stages: First, we reference prior work where we identified and defined the Information Requirements (IRs) and Functional Requirements (FRs) for PMx, forming the blueprint for a unified framework. Second, we conduct a literature review to assess current DT applications integrating these IRs and FRs, revealing standardized DT models and tools that support automated PMx. Lastly, we highlight gaps in current DT implementations, particularly those IRs and FRs not fully supported, and outline the necessary components for a comprehensive, automated PMx system. Our paper concludes with research directions aimed at seamlessly integrating DTs into the PMx paradigm to achieve this ambitious vision.
       


### [Enhancing supply chain security with automated machine learning](https://arxiv.org/abs/2406.13166)

**Authors:**
Haibo Wang, Lutfu S. Sua, Bahram Alidaee

**Abstract:**
The increasing scale and complexity of global supply chains have led to new challenges spanning various fields, such as supply chain disruptions due to long waiting lines at the ports, material shortages, and inflation. Coupled with the size of supply chains and the availability of vast amounts of data, efforts towards tackling such challenges have led to an increasing interest in applying machine learning methods in many aspects of supply chains. Unlike other solutions, ML techniques, including Random Forest, XGBoost, LightGBM, and Neural Networks, make predictions and approximate optimal solutions faster. This paper presents an automated ML framework to enhance supply chain security by detecting fraudulent activities, predicting maintenance needs, and forecasting material backorders. Using datasets of varying sizes, results show that fraud detection achieves an 88% accuracy rate using sampling methods, machine failure prediction reaches 93.4% accuracy, and material backorder prediction achieves 89.3% accuracy. Hyperparameter tuning significantly improved the performance of these models, with certain supervised techniques like XGBoost and LightGBM reaching up to 100% precision. This research contributes to supply chain security by streamlining data preprocessing, feature selection, model optimization, and inference deployment, addressing critical challenges and boosting operational efficiency.
       


### [Remaining useful life prediction of rolling bearings based on refined composite multi-scale attention entropy and dispersion entropy](https://arxiv.org/abs/2406.16967)

**Authors:**
Yunchong Long, Qinkang Pang, Guangjie Zhu, Junxian Cheng, Xiangshun Li

**Abstract:**
Remaining useful life (RUL) prediction based on vibration signals is crucial for ensuring the safe operation and effective health management of rotating machinery. Existing studies often extract health indicators (HI) from time domain and frequency domain features to analyze complex vibration signals, but these features may not accurately capture the degradation process. In this study, we propose a degradation feature extraction method called Fusion of Multi-Modal Multi-Scale Entropy (FMME), which utilizes multi-modal Refined Composite Multi-scale Attention Entropy (RCMATE) and Fluctuation Dispersion Entropy (RCMFDE), to solve the problem that the existing degradation features cannot accurately reflect the degradation process. Firstly, the Empirical Mode Decomposition (EMD) is employed to decompose the dual-channel vibration signals of bearings into multiple modals. The main modals are then selected for further analysis. The subsequent step involves the extraction of RCMATE and RCMFDE from each modal, followed by wavelet denoising. Next, a novel metric is proposed to evaluate the quality of degradation features. The attention entropy and dispersion entropy of the optimal scales under different modals are fused using Laplacian Eigenmap (LE) to obtain the health indicators. Finally, RUL prediction is performed through the similarity of health indicators between fault samples and bearings to be predicted. Experimental results demonstrate that the proposed method yields favorable outcomes across diverse operating conditions.
       


### [Integrating Generative AI with Network Digital Twins for Enhanced Network Operations](https://arxiv.org/abs/2406.17112)

**Authors:**
Kassi Muhammad, Teef David, Giulia Nassisid, Tina Farus

**Abstract:**
As telecommunications networks become increasingly complex, the integration of advanced technologies such as network digital twins and generative artificial intelligence (AI) emerges as a pivotal solution to enhance network operations and resilience. This paper explores the synergy between network digital twins, which provide a dynamic virtual representation of physical networks, and generative AI, particularly focusing on Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). We propose a novel architectural framework that incorporates these technologies to significantly improve predictive maintenance, network scenario simulation, and real-time data-driven decision-making. Through extensive simulations, we demonstrate how generative AI can enhance the accuracy and operational efficiency of network digital twins, effectively handling real-world complexities such as unpredictable traffic loads and network failures. The findings suggest that this integration not only boosts the capability of digital twins in scenario forecasting and anomaly detection but also facilitates a more adaptive and intelligent network management system.
       


## July
### [Learning operando impedance function for battery health with aging-aware equivalent circuit model](https://arxiv.org/abs/2407.06639)

**Authors:**
Zihao Zhou, Antti Aitio, David Howey

**Abstract:**
The wide usage of Lithium-ion batteries (LIBs) requires a deep understanding about battery health. Estimation of battery state-of-health (SOH) is a crucial but yet still challenging task. Pure model-based methods may suffer from inaccuracy and instability of parameter estimations in lifelong aging. While pure data-driven methods rely heavily on quality and quantity of training set, causing lack of generality when extrapolating into unseen cases. In this paper, we propose an aging-aware equivalent circuit model (ECM), which combines model-based and data-driven techniques for SOH estimation. Gaussian process (GP) regression is incorporated in ECM to modelling parameters dependency on operating condition and aging time. The state space formulation of GP is used to enable a computationally efficient co-estimation framework of both parameters and states. Samples from two Open datasets are used to validate model performance, which gives accurate estimation for capacity and impedance. The learnt impedance function can be lined to the shape change of open circuit voltage (OCV) versus SOC curve and thus providing a way to further estimate changes of differential voltage (dV/dQ) curves.
       


### [Cost-optimized probabilistic maintenance for condition monitoring of wind turbines with rare failures](https://arxiv.org/abs/2407.09385)

**Authors:**
Viktor Begun, Ulrich Schlickewei

**Abstract:**
We propose a method, a model, and a form of presenting model results for condition monitoring of a small set of wind turbines with rare failures. The main new ingredient of the method is to sample failure thresholds according to the profit they give to an operating company. The model is a multiple linear regression with seasonal components and external regressors, representing all sensor components except for the considered one. To overcome the scarcity of the training data, we use the median sensor values from all available turbines in their healthy state. The cumulated deviation from the normal behavior model obtained for this median turbine is calibrated for each turbine at the beginning of the test period and after known failures. The proposed form of presenting results is to set a scale for possible costs, control for random maintenance, and show a whole distribution of costs depending on the free model parameters. We make a case study on an open dataset with SCADA data from multiple sensors and show that considering the influence of turbine components is more critical than seasonality. The distribution, the average, and the standard deviation of maintenance costs can be very different for similar minimal costs. Random maintenance can be more profitable than reactive maintenance and other approaches. Our predictive maintenance model outperforms random maintenance and competitors for the whole set of considered turbines, giving substantial savings.
       


### [Application of cloud computing platform in industrial big data processing](https://arxiv.org/abs/2407.09491)

**Author:**
Ziyan Yao

**Abstract:**
With the rapid growth and increasing complexity of industrial big data, traditional data processing methods are facing many challenges. This article takes an in-depth look at the application of cloud computing technology in industrial big data processing and explores its potential impact on improving data processing efficiency, security, and cost-effectiveness. The article first reviews the basic principles and key characteristics of cloud computing technology, and then analyzes the characteristics and processing requirements of industrial big data. In particular, this study focuses on the application of cloud computing in real-time data processing, predictive maintenance, and optimization, and demonstrates its practical effects through case studies. At the same time, this article also discusses the main challenges encountered during the implementation process, such as data security, privacy protection, performance and scalability issues, and proposes corresponding solution strategies. Finally, this article looks forward to the future trends of the integration of cloud computing and industrial big data, as well as the application prospects of emerging technologies such as artificial intelligence and machine learning in this field. The results of this study not only provide practical guidance for cloud computing applications in the industry, but also provide a basis for further research in academia.
       


### [XEdgeAI: A Human-centered Industrial Inspection Framework with Data-centric Explainable Edge AI Approach](https://arxiv.org/abs/2407.11771)

**Authors:**
Truong Thanh Hung Nguyen, Phuc Truong Loc Nguyen, Hung Cao

**Abstract:**
Recent advancements in deep learning have significantly improved visual quality inspection and predictive maintenance within industrial settings. However, deploying these technologies on low-resource edge devices poses substantial challenges due to their high computational demands and the inherent complexity of Explainable AI (XAI) methods. This paper addresses these challenges by introducing a novel XAI-integrated Visual Quality Inspection framework that optimizes the deployment of semantic segmentation models on low-resource edge devices. Our framework incorporates XAI and the Large Vision Language Model to deliver human-centered interpretability through visual and textual explanations to end-users. This is crucial for end-user trust and model interpretability. We outline a comprehensive methodology consisting of six fundamental modules: base model fine-tuning, XAI-based explanation generation, evaluation of XAI approaches, XAI-guided data augmentation, development of an edge-compatible model, and the generation of understandable visual and textual explanations. Through XAI-guided data augmentation, the enhanced model incorporating domain expert knowledge with visual and textual explanations is successfully deployed on mobile devices to support end-users in real-world scenarios. Experimental results showcase the effectiveness of the proposed framework, with the mobile model achieving competitive accuracy while significantly reducing model size. This approach paves the way for the broader adoption of reliable and interpretable AI tools in critical industrial applications, where decisions must be both rapid and justifiable. Our code for this work can be found at https://github.com/Analytics-Everywhere-Lab/vqixai.
       


### [SOC-Boundary and Battery Aging Aware Hierarchical Coordination of Multiple EV Aggregates Among Multi-stakeholders with Multi-Agent Constrained Deep Reinforcement Learning](https://arxiv.org/abs/2407.13790)

**Author:**
Xin Chen

**Abstract:**
As electric vehicles (EV) become more prevalent and advances in electric vehicle electronics continue, vehicle-to-grid (V2G) techniques and large-scale scheduling strategies are increasingly important to promote renewable energy utilization and enhance the stability of the power grid. This study proposes a hierarchical multistakeholder V2G coordination strategy based on safe multi-agent constrained deep reinforcement learning (MCDRL) and the Proof-of-Stake algorithm to optimize benefits for all stakeholders, including the distribution system operator (DSO), electric vehicle aggregators (EVAs) and EV users. For DSO, the strategy addresses load fluctuations and the integration of renewable energy. For EVAs, energy constraints and charging costs are considered. The three critical parameters of battery conditioning, state of charge (SOC), state of power (SOP), and state of health (SOH), are crucial to the participation of EVs in V2G. Hierarchical multi-stakeholder V2G coordination significantly enhances the integration of renewable energy, mitigates load fluctuations, meets the energy demands of the EVAs, and reduces charging costs and battery degradation simultaneously.
       


### [Transformer-based Capacity Prediction for Lithium-ion Batteries with Data Augmentation](https://arxiv.org/abs/2407.16036)

**Authors:**
Gift Modekwe, Saif Al-Wahaibi, Qiugang Lu

**Abstract:**
Lithium-ion batteries are pivotal to technological advancements in transportation, electronics, and clean energy storage. The optimal operation and safety of these batteries require proper and reliable estimation of battery capacities to monitor the state of health. Current methods for estimating the capacities fail to adequately account for long-term temporal dependencies of key variables (e.g., voltage, current, and temperature) associated with battery aging and degradation. In this study, we explore the usage of transformer networks to enhance the estimation of battery capacity. We develop a transformer-based battery capacity prediction model that accounts for both long-term and short-term patterns in battery data. Further, to tackle the data scarcity issue, data augmentation is used to increase the data size, which helps to improve the performance of the model. Our proposed method is validated with benchmark datasets. Simulation results show the effectiveness of data augmentation and the transformer network in improving the accuracy and robustness of battery capacity prediction.
       


### [Generative Learning for Simulation of Vehicle Faults](https://arxiv.org/abs/2407.17654)

**Authors:**
Patrick Kuiper, Sirui Lin, Jose Blanchet, Vahid Tarokh

**Abstract:**
We develop a novel generative model to simulate vehicle health and forecast faults, conditioned on practical operational considerations. The model, trained on data from the US Army's Predictive Logistics program, aims to support predictive maintenance. It forecasts faults far enough in advance to execute a maintenance intervention before a breakdown occurs. The model incorporates real-world factors that affect vehicle health. It also allows us to understand the vehicle's condition by analyzing operating data, and characterizing each vehicle into discrete states. Importantly, the model predicts the time to first fault with high accuracy. We compare its performance to other models and demonstrate its successful training.
       


### [Graph Neural Networks for Virtual Sensing in Complex Systems: Addressing Heterogeneous Temporal Dynamics](https://arxiv.org/abs/2407.18691)

**Authors:**
Mengjie Zhao, Cees Taal, Stephan Baggerohr, Olga Fink

**Abstract:**
Real-time condition monitoring is crucial for the reliable and efficient operation of complex systems. However, relying solely on physical sensors can be limited due to their cost, placement constraints, or inability to directly measure certain critical parameters. Virtual sensing addresses these limitations by leveraging readily available sensor data and system knowledge to estimate inaccessible parameters or infer system states. The increasing complexity of industrial systems necessitates deployments of sensors with diverse modalities to provide a comprehensive understanding of system states. These sensors capture data at varying frequencies to monitor both rapid and slowly varying system dynamics, as well as local and global state evolutions of the systems. This leads to heterogeneous temporal dynamics, which, particularly under varying operational end environmental conditions, pose a significant challenge for accurate virtual sensing. To address this, we propose a Heterogeneous Temporal Graph Neural Network (HTGNN) framework. HTGNN explicitly models signals from diverse sensors and integrates operating conditions into the model architecture. We evaluate HTGNN using two newly released datasets: a bearing dataset with diverse load conditions for bearing load prediction and a year-long simulated dataset for predicting bridge live loads. Our results demonstrate that HTGNN significantly outperforms established baseline methods in both tasks, particularly under highly varying operating conditions. These results highlight HTGNN's potential as a robust and accurate virtual sensing approach for complex systems, paving the way for improved monitoring, predictive maintenance, and enhanced system performance.
       


### [Identifying fatigue crack initiation through analytical calculation of temporal compliance calibrated with Computed Tomography](https://arxiv.org/abs/2407.20245)

**Authors:**
Ritam Pal, Amrita Basak

**Abstract:**
Fatigue failure is ubiquitous in engineering applications. While the total fatigue life is critical to understanding a component's operational life, for safety, regulatory compliance, and predictive maintenance, the characterization of initiation life is important. Traditionally, initiation life is characterized by potential drop method, acoustic emission technique, and strain-based measurements. However, the primary challenge with these methods lies in the necessity of calibration for each new material system. The difficulties become even more aggravated for additively manufactured components, where fatigue properties are reported to vary widely in the open literature. In this work, an analytical methodology is utilized to evaluate the initiation life of two different materials such as AlSi10Mg and SS316L, fabricated via laser-powder bed fusion (L-PBF) technique. The processing parameters are selected such that AlSi10Mg behaves like a brittle material while SS316L shows ductile behavior. A custom fatigue testing apparatus is used inside Computed Tomography (CT) for evaluating fatigue initiation. The apparatus reports load-displacement data, which is post-processed using an analytical approach to calculate the evolution of material compliance. The results indicate that crack initiation during fatigue loading is marked by a noticeable change in compliance. The analytical technique shows a maximum difference of 4.8% in predicting initiation life compared to CT imaging. These findings suggest that compliance monitoring can effectively identify fatigue initiation in various materials.
       


### [Industrial-Grade Smart Troubleshooting through Causal Technical Language Processing: a Proof of Concept](https://arxiv.org/abs/2407.20700)

**Authors:**
Alexandre Trilla, Ossee Yiboe, Nenad Mijatovic, Jordi Vitrià

**Abstract:**
This paper describes the development of a causal diagnosis approach for troubleshooting an industrial environment on the basis of the technical language expressed in Return on Experience records. The proposed method leverages the vectorized linguistic knowledge contained in the distributed representation of a Large Language Model, and the causal associations entailed by the embedded failure modes and mechanisms of the industrial assets. The paper presents the elementary but essential concepts of the solution, which is conceived as a causality-aware retrieval augmented generation system, and illustrates them experimentally on a real-world Predictive Maintenance setting. Finally, it discusses avenues of improvement for the maturity of the utilized causal technology to meet the robustness challenges of increasingly complex scenarios in the industry.
       


## August
### [Low-Power Vibration-Based Predictive Maintenance for Industry 4.0 using Neural Networks: A Survey](https://arxiv.org/abs/2408.00516)

**Authors:**
Alexandru Vasilache, Sven Nitzsche, Daniel Floegel, Tobias Schuermann, Stefan von Dosky, Thomas Bierweiler, Marvin Mußler, Florian Kälber, Soeren Hohmann, Juergen Becker

**Abstract:**
The advancements in smart sensors for Industry 4.0 offer ample opportunities for low-powered predictive maintenance and condition monitoring. However, traditional approaches in this field rely on processing in the cloud, which incurs high costs in energy and storage. This paper investigates the potential of neural networks for low-power on-device computation of vibration sensor data for predictive maintenance. We review the literature on Spiking Neural Networks (SNNs) and Artificial Neuronal Networks (ANNs) for vibration-based predictive maintenance by analyzing datasets, data preprocessing, network architectures, and hardware implementations. Our findings suggest that no satisfactory standard benchmark dataset exists for evaluating neural networks in predictive maintenance tasks. Furthermore frequency domain transformations are commonly employed for preprocessing. SNNs mainly use shallow feed forward architectures, whereas ANNs explore a wider range of models and deeper networks. Finally, we highlight the need for future research on hardware implementations of neural networks for low-power predictive maintenance applications and the development of a standardized benchmark dataset.
       


### [Relax, Estimate, and Track: a Simple Battery State-of-charge and State-of-health Estimation Method](https://arxiv.org/abs/2408.01127)

**Authors:**
Shida Jiang, Junzhe Shi, Scott Moura

**Abstract:**
Battery management is a critical component of ubiquitous battery-powered energy systems, in which battery state-of-charge (SOC) and state-of-health (SOH) estimations are of crucial importance. Conventional SOC and SOH estimation methods, especially model-based methods, often lack accurate modeling of the open circuit voltage (OCV), have relatively high computational complexity, and lack theoretical analysis. This study introduces a simple SOC and SOH estimation method that overcomes all these weaknesses. The key idea of the proposed method is to momentarily set the cell's current to zero for a few minutes during the charging, perform SOC and SOH estimation based on the measured data, and continue tracking the cell's SOC afterward. The method is based on rigorous theoretical analysis, requires no hyperparameter fine-tuning, and is hundreds of times faster than conventional model-based methods. The method is validated on six batteries charged at different C rates and temperatures, realizing fast and accurate estimations under various conditions, with a SOH root mean square error (RMSE) of around 3% and a SOC RMSE of around 1.5%.
       


### [Performance Classification and Remaining Useful Life Prediction of Lithium Batteries Using Machine Learning and Early Cycle Electrochemical Impedance Spectroscopy Measurements](https://arxiv.org/abs/2408.03469)

**Authors:**
Christian Parsons, Adil Amin, Prasenjit Guptasarma

**Abstract:**
We presents an approach for early cycle classification of lithium-ion batteries into high and low-performing categories, coupled with the prediction of their remaining useful life (RUL) using a linear lasso technique. Traditional methods often rely on extensive cycling and the measurement of a large number of electrochemical impedance spectroscopy (EIS) frequencies to assess battery performance, which can be time and resource consuming. In this study, we propose a methodology that leverages specific EIS frequencies to achieve accurate classification and RUL prediction within the first few cycles of battery operation. Notably, given only the 20 kHz impedance response, our support vector machine (SVM) model classifies batteries with 100\% accuracy. Additionally, our findings reveal that battery performance classification is frequency agnostic within the high frequency ($<20$ kHz) to low-frequency (32 mHz) range. Our model also demonstrates accurate RUL predictions with $R^2>0.96$ based on the out of phase impedance response at a single high (20 kHz) and a single mid-frequency (8.8 Hz), in conjunction with temperature data. This research underscores the significance of the mid-frequency impedance response as merely one among several crucial features in determining battery performance, thereby broadening the understanding of factors influencing battery behavior.
       


### [Predictive maintenance solution for industrial systems -- an unsupervised approach based on log periodic power law](https://arxiv.org/abs/2408.05231)

**Author:**
Bogdan Łobodziński

**Abstract:**
A new unsupervised predictive maintenance analysis method based on the renormalization group approach used to discover critical behavior in complex systems has been proposed. The algorithm analyzes univariate time series and detects critical points based on a newly proposed theorem that identifies critical points using a Log Periodic Power Law function fits. Application of a new algorithm for predictive maintenance analysis of industrial data collected from reciprocating compressor systems is presented. Based on the knowledge of the dynamics of the analyzed compressor system, the proposed algorithm predicts valve and piston rod seal failures well in advance.
       


### [A Digital Twin Framework Utilizing Machine Learning for Robust Predictive Maintenance: Enhancing Tire Health Monitoring](https://arxiv.org/abs/2408.06220)

**Authors:**
Vispi Karkaria, Jie Chen, Christopher Luey, Chase Siuta, Damien Lim, Robert Radulescu, Wei Chen

**Abstract:**
We introduce a novel digital twin framework for predictive maintenance of long-term physical systems. Using monitoring tire health as an application, we show how the digital twin framework can be used to enhance automotive safety and efficiency, and how the technical challenges can be overcome using a three-step approach. Firstly, for managing the data complexity over a long operation span, we employ data reduction techniques to concisely represent physical tires using historical performance and usage data. Relying on these data, for fast real-time prediction, we train a transformer-based model offline on our concise dataset to predict future tire health over time, represented as Remaining Casing Potential (RCP). Based on our architecture, our model quantifies both epistemic and aleatoric uncertainty, providing reliable confidence intervals around predicted RCP. Secondly, to incorporate real-time data, we update the predictive model in the digital twin framework, ensuring its accuracy throughout its life span with the aid of hybrid modeling and the use of discrepancy function. Thirdly, to assist decision making in predictive maintenance, we implement a Tire State Decision Algorithm, which strategically determines the optimal timing for tire replacement based on RCP forecasted by our transformer model. This approach ensures our digital twin accurately predicts system health, continually refines its digital representation, and supports predictive maintenance decisions. Our framework effectively embodies a physical system, leveraging big data and machine learning for predictive maintenance, model updates, and decision-making.
       


### [Battery GraphNets : Relational Learning for Lithium-ion Batteries(LiBs) Life Estimation](https://arxiv.org/abs/2408.07624)

**Authors:**
Sakhinana Sagar Srinivas, Rajat Kumar Sarkar, Venkataramana Runkana

**Abstract:**
Battery life estimation is critical for optimizing battery performance and guaranteeing minimal degradation for better efficiency and reliability of battery-powered systems. The existing methods to predict the Remaining Useful Life(RUL) of Lithium-ion Batteries (LiBs) neglect the relational dependencies of the battery parameters to model the nonlinear degradation trajectories. We present the Battery GraphNets framework that jointly learns to incorporate a discrete dependency graph structure between battery parameters to capture the complex interactions and the graph-learning algorithm to model the intrinsic battery degradation for RUL prognosis. The proposed method outperforms several popular methods by a significant margin on publicly available battery datasets and achieves SOTA performance. We report the ablation studies to support the efficacy of our approach.
       


### [Augmenting train maintenance technicians with automated incident diagnostic suggestions](https://arxiv.org/abs/2408.10288)

**Authors:**
Georges Tod, Jean Bruggeman, Evert Bevernage, Pieter Moelans, Walter Eeckhout, Jean-Luc Glineur

**Abstract:**
Train operational incidents are so far diagnosed individually and manually by train maintenance technicians. In order to assist maintenance crews in their responsiveness and task prioritization, a learning machine is developed and deployed in production to suggest diagnostics to train technicians on their phones, tablets or laptops as soon as a train incident is declared. A feedback loop allows to take into account the actual diagnose by designated train maintenance experts to refine the learning machine. By formulating the problem as a discrete set classification task, feature engineering methods are proposed to extract physically plausible sets of events from traces generated on-board railway vehicles. The latter feed an original ensemble classifier to class incidents by their potential technical cause. Finally, the resulting model is trained and validated using real operational data and deployed on a cloud platform. Future work will explore how the extracted sets of events can be used to avoid incidents by assisting human experts in the creation predictive maintenance alerts.
       


### [Finding the DeepDream for Time Series: Activation Maximization for Univariate Time Series](https://arxiv.org/abs/2408.10628)

**Authors:**
Udo Schlegel, Daniel A. Keim, Tobias Sutter

**Abstract:**
Understanding how models process and interpret time series data remains a significant challenge in deep learning to enable applicability in safety-critical areas such as healthcare. In this paper, we introduce Sequence Dreaming, a technique that adapts Activation Maximization to analyze sequential information, aiming to enhance the interpretability of neural networks operating on univariate time series. By leveraging this method, we visualize the temporal dynamics and patterns most influential in model decision-making processes. To counteract the generation of unrealistic or excessively noisy sequences, we enhance Sequence Dreaming with a range of regularization techniques, including exponential smoothing. This approach ensures the production of sequences that more accurately reflect the critical features identified by the neural network. Our approach is tested on a time series classification dataset encompassing applications in predictive maintenance. The results show that our proposed Sequence Dreaming approach demonstrates targeted activation maximization for different use cases so that either centered class or border activation maximization can be generated. The results underscore the versatility of Sequence Dreaming in uncovering salient temporal features learned by neural networks, thereby advancing model transparency and trustworthiness in decision-critical domains.
       


### [Explainable Anomaly Detection: Counterfactual driven What-If Analysis](https://arxiv.org/abs/2408.11935)

**Authors:**
Logan Cummins, Alexander Sommers, Sudip Mittal, Shahram Rahimi, Maria Seale, Joseph Jaboure, Thomas Arnold

**Abstract:**
There exists three main areas of study inside of the field of predictive maintenance: anomaly detection, fault diagnosis, and remaining useful life prediction. Notably, anomaly detection alerts the stakeholder that an anomaly is occurring. This raises two fundamental questions: what is causing the fault and how can we fix it? Inside of the field of explainable artificial intelligence, counterfactual explanations can give that information in the form of what changes to make to put the data point into the opposing class, in this case "healthy". The suggestions are not always actionable which may raise the interest in asking "what if we do this instead?" In this work, we provide a proof of concept for utilizing counterfactual explanations as what-if analysis. We perform this on the PRONOSTIA dataset with a temporal convolutional network as the anomaly detector. Our method presents the counterfactuals in the form of a what-if analysis for this base problem to inspire future work for more complex systems and scenarios.
       


### [Synergistic and Efficient Edge-Host Communication for Energy Harvesting Wireless Sensor Networks](https://arxiv.org/abs/2408.14379)

**Authors:**
Cyan Subhra Mishra, Jack Sampson, Mahmut Taylan Kandmeir, Vijaykrishnan Narayanan, Chita R Das

**Abstract:**
There is an increasing demand for intelligent processing on ultra-low-power internet of things (IoT) device. Recent works have shown substantial efficiency boosts by executing inferences directly on the IoT device (node) rather than transmitting data. However, the computation and power demands of Deep Neural Network (DNN)-based inference pose significant challenges in an energy-harvesting wireless sensor network (EH-WSN). Moreover, these tasks often require responses from multiple physically distributed EH sensor nodes, which impose crucial system optimization challenges in addition to per-node constraints. To address these challenges, we propose Seeker, a hardware-software co-design approach for increasing on-sensor computation, reducing communication volume, and maximizing inference completion, without violating the quality of service, in EH-WSNs coordinated by a mobile device. Seeker uses a store-and-execute approach to complete a subset of inferences on the EH sensor node, reducing communication with the mobile host. Further, for those inferences unfinished because of the harvested energy constraints, it leverages task-aware coreset construction to efficiently communicate compact features to the host device. We evaluate Seeker for human activity recognition, as well as predictive maintenance and show ~8.9x reduction in communication data volume with 86.8% accuracy, surpassing the 81.2% accuracy of the state-of-the-art.
       


### [Vibration Sensor Dataset for Estimating Fan Coil Motor Health](https://arxiv.org/abs/2408.14448)

**Authors:**
Heitor Lifsitch, Gabriel Rocha, Hendrio Bragança, Cláudio Filho, Leandro Okimoto, Allan Amorin, Fábio Cardoso

**Abstract:**
To enhance the field of continuous motor health monitoring, we present FAN-COIL-I, an extensive vibration sensor dataset derived from a Fan Coil motor. This dataset is uniquely positioned to facilitate the detection and prediction of motor health issues, enabling a more efficient maintenance scheduling process that can potentially obviate the need for regular checks. Unlike existing datasets, often created under controlled conditions or through simulations, FAN-COIL-I is compiled from real-world operational data, providing an invaluable resource for authentic motor diagnosis and predictive maintenance research. Gathered using a high-resolution 32KHz sampling rate, the dataset encompasses comprehensive vibration readings from both the forward and rear sides of the Fan Coil motor over a continuous two-week period, offering a rare glimpse into the dynamic operational patterns of these systems in a corporate setting. FAN-COIL-I stands out not only for its real-world applicability but also for its potential to serve as a reliable benchmark for researchers and practitioners seeking to validate their models against genuine engine conditions.
       


### [Anomaly Detection in Time Series of EDFA Pump Currents to Monitor Degeneration Processes using Fuzzy Clustering](https://arxiv.org/abs/2408.15268)

**Authors:**
Dominic Schneider, Lutz Rapp, Christoph Ament

**Abstract:**
This article proposes a novel fuzzy clustering based anomaly detection method for pump current time series of EDFA systems. The proposed change detection framework (CDF) strategically combines the advantages of entropy analysis (EA) and principle component analysis (PCA) with fuzzy clustering procedures. In the framework, EA is applied for dynamic selection of features for reduction of the feature space and increase of computational performance. Furthermore, PCA is utilized to extract features from the raw feature space to enable generalization capability of the subsequent fuzzy clustering procedures. Three different fuzzy clustering methods, more precisely the fuzzy clustering algorithm, a probabilistic clustering algorithm and a possibilistic clustering algorithm are evaluated for performance and generalization. Hence, the proposed framework has the innovative feature to detect changes in pump current time series at an early stage for arbitrary points of operation, compared to state-of-the-art predefined alarms in commercially used EDFAs. Moreover, the approach is implemented and tested using experimental data. In addition, the proposed framework enables further approaches of applying decentralized predictive maintenance for optical fiber networks.
       


### [The NIRSpec Micro-Shutter Array: Operability and Operations After Two Years of JWST Science](https://arxiv.org/abs/2408.15940)

**Authors:**
Katie Bechtold, Torsten Böker, David E. Franz, Maurice te Plate, Timothy D. Rawle, Rai Wu, Peter Zeidler

**Abstract:**
The Near Infrared Spectrograph (NIRSpec) on the James Webb Space Telescope affords the astronomical community an unprecedented space-based Multi-Object Spectroscopy (MOS) capability through the use of a programmable array of micro-electro-mechanical shutters. Launched in December 2021 and commissioned along with a suite of other observatory instruments throughout the first half of 2022, NIRSpec has been carrying out scientific observations since the completion of commissioning. These observations would not be possible without a rigorous program of engineering operations to actively monitor and maintain NIRSpec's hardware health and safety and enhance instrument efficiency and performance. Although MOS is only one of the observing modes available to users, the complexity and uniqueness of the Micro-Shutter Assembly (MSA) that enables it has presented a variety of engineering challenges, including the appearance of electrical shorts that produce contaminating glow in exposures. Despite these challenges, the NIRSpec Multi-Object Spectrograph continues to perform robustly with no discernible degradation or significant reduction in capability.
  This paper provides an overview of the NIRSpec micro-shutter subsystem's state of health and operability and presents some of the developments that have taken place in its operation since the completion of instrument commissioning.
       


### [Economic Optimal Power Management of Second-Life Battery Energy Storage Systems](https://arxiv.org/abs/2408.16197)

**Authors:**
Amir Farakhor, Di Wu, Pingen Chen, Junmin Wang, Yebin Wang, Huazhen Fang

**Abstract:**
Second-life battery energy storage systems (SL-BESS) are an economical means of long-duration grid energy storage. They utilize retired battery packs from electric vehicles to store and provide electrical energy at the utility scale. However, they pose critical challenges in achieving optimal utilization and extending their remaining useful life. These complications primarily result from the constituent battery packs' inherent heterogeneities in terms of their size, chemistry, and degradation. This paper proposes an economic optimal power management approach to ensure the cost-minimized operation of SL-BESS while adhering to safety regulations and maintaining a balance between the power supply and demand. The proposed approach takes into account the costs associated with the degradation, energy loss, and decommissioning of the battery packs. In particular, we capture the degradation costs of the retired battery packs through a weighted average Ah-throughput aging model. The presented model allows us to quantify the capacity fading for second-life battery packs for different operating temperatures and C-rates. To evaluate the performance of the proposed approach, we conduct extensive simulations on a SL-BESS consisting of various heterogeneous retired battery packs in the context of grid operation. The results offer novel insights into SL-BESS operation and highlight the importance of prudent power management to ensure economically optimal utilization.
       


## September
### [Graph neural network-based lithium-ion battery state of health estimation using partial discharging curve](https://arxiv.org/abs/2409.00141)

**Authors:**
Kate Qi Zhou, Yan Qin, Chau Yuen

**Abstract:**
Data-driven methods have gained extensive attention in estimating the state of health (SOH) of lithium-ion batteries. Accurate SOH estimation requires degradation-relevant features and alignment of statistical distributions between training and testing datasets. However, current research often overlooks these needs and relies on arbitrary voltage segment selection. To address these challenges, this paper introduces an innovative approach leveraging spatio-temporal degradation dynamics via graph convolutional networks (GCNs). Our method systematically selects discharge voltage segments using the Matrix Profile anomaly detection algorithm, eliminating the need for manual selection and preventing information loss. These selected segments form a fundamental structure integrated into the GCN-based SOH estimation model, capturing inter-cycle dynamics and mitigating statistical distribution incongruities between offline training and online testing data. Validation with a widely accepted open-source dataset demonstrates that our method achieves precise SOH estimation, with a root mean squared error of less than 1%.
       


### [DefectTwin: When LLM Meets Digital Twin for Railway Defect Inspection](https://arxiv.org/abs/2409.06725)

**Authors:**
Rahatara Ferdousi, M. Anwar Hossain, Chunsheng Yang, Abdulmotaleb El Saddik

**Abstract:**
A Digital Twin (DT) replicates objects, processes, or systems for real-time monitoring, simulation, and predictive maintenance. Recent advancements like Large Language Models (LLMs) have revolutionized traditional AI systems and offer immense potential when combined with DT in industrial applications such as railway defect inspection. Traditionally, this inspection requires extensive defect samples to identify patterns, but limited samples can lead to overfitting and poor performance on unseen defects. Integrating pre-trained LLMs into DT addresses this challenge by reducing the need for vast sample data. We introduce DefectTwin, which employs a multimodal and multi-model (M^2) LLM-based AI pipeline to analyze both seen and unseen visual defects in railways. This application enables a railway agent to perform expert-level defect analysis using consumer electronics (e.g., tablets). A multimodal processor ensures responses are in a consumable format, while an instant user feedback mechanism (instaUF) enhances Quality-of-Experience (QoE). The proposed M^2 LLM outperforms existing models, achieving high precision (0.76-0.93) across multimodal inputs including text, images, and videos of pre-trained defects, and demonstrates superior zero-shot generalizability for unseen defects. We also evaluate the latency, token count, and usefulness of responses generated by DefectTwin on consumer devices. To our knowledge, DefectTwin is the first LLM-integrated DT designed for railway defect inspection.
       


### [A Continual and Incremental Learning Approach for TinyML On-device Training Using Dataset Distillation and Model Size Adaption](https://arxiv.org/abs/2409.07114)

**Authors:**
Marcus Rüb, Philipp Tuchel, Axel Sikora, Daniel Mueller-Gritschneder

**Abstract:**
A new algorithm for incremental learning in the context of Tiny Machine learning (TinyML) is presented, which is optimized for low-performance and energy efficient embedded devices. TinyML is an emerging field that deploys machine learning models on resource-constrained devices such as microcontrollers, enabling intelligent applications like voice recognition, anomaly detection, predictive maintenance, and sensor data processing in environments where traditional machine learning models are not feasible. The algorithm solve the challenge of catastrophic forgetting through the use of knowledge distillation to create a small, distilled dataset. The novelty of the method is that the size of the model can be adjusted dynamically, so that the complexity of the model can be adapted to the requirements of the task. This offers a solution for incremental learning in resource-constrained environments, where both model size and computational efficiency are critical factors. Results show that the proposed algorithm offers a promising approach for TinyML incremental learning on embedded devices. The algorithm was tested on five datasets including: CIFAR10, MNIST, CORE50, HAR, Speech Commands. The findings indicated that, despite using only 43% of Floating Point Operations (FLOPs) compared to a larger fixed model, the algorithm experienced a negligible accuracy loss of just 1%. In addition, the presented method is memory efficient. While state-of-the-art incremental learning is usually very memory intensive, the method requires only 1% of the original data set.
       


### [Parallel Reduced Order Modeling for Digital Twins using High-Performance Computing Workflows](https://arxiv.org/abs/2409.09080)

**Authors:**
S. Ares de Parga, J. R. Bravo, N. Sibuet, J. A. Hernandez, R. Rossi, Stefan Boschert, Enrique S. Quintana-Ortí, Andrés E. Tomás, Cristian Cătălin Tatu, Fernando Vázquez-Novoa, Jorge Ejarque, Rosa M. Badia

**Abstract:**
The integration of Reduced Order Models (ROMs) with High-Performance Computing (HPC) is critical for developing digital twins, particularly for real-time monitoring and predictive maintenance of industrial systems. This paper describes a comprehensive, HPC-enabled workflow for developing and deploying projection-based ROMs (PROMs). We use PyCOMPSs' parallel framework to efficiently execute ROM training simulations, employing parallel Singular Value Decomposition (SVD) algorithms such as randomized SVD, Lanczos SVD, and full SVD based on Tall-Skinny QR. In addition, we introduce a partitioned version of the hyper-reduction scheme known as the Empirical Cubature Method. Despite the widespread use of HPC for PROMs, there is a significant lack of publications detailing comprehensive workflows for building and deploying end-to-end PROMs in HPC environments. Our workflow is validated through a case study focusing on the thermal dynamics of a motor. The PROM is designed to deliver a real-time prognosis tool that could enable rapid and safe motor restarts post-emergency shutdowns under different operating conditions for further integration into digital twins or control systems. To facilitate deployment, we use the HPC Workflow as a Service strategy and Functional Mock-Up Units to ensure compatibility and ease of integration across HPC, edge, and cloud environments. The outcomes illustrate the efficacy of combining PROMs and HPC, establishing a precedent for scalable, real-time digital twin applications across multiple industries.
       


### [Fault Analysis And Predictive Maintenance Of Induction Motor Using Machine Learning](https://arxiv.org/abs/2409.09944)

**Authors:**
Kavana Venkatesh, Neethi M

**Abstract:**
Induction motors are one of the most crucial electrical equipment and are extensively used in industries in a wide range of applications. This paper presents a machine learning model for the fault detection and classification of induction motor faults by using three phase voltages and currents as inputs. The aim of this work is to protect vital electrical components and to prevent abnormal event progression through early detection and diagnosis. This work presents a fast forward artificial neural network model to detect some of the commonly occurring electrical faults like overvoltage, under voltage, single phasing, unbalanced voltage, overload, ground fault. A separate model free monitoring system wherein the motor itself acts like a sensor is presented and the only monitored signals are the input given to the motor. Limits for current and voltage values are set for the faulty and healthy conditions, which is done by a classifier. Real time data from a 0.33 HP induction motor is used to train and test the neural network. The model so developed analyses the voltage and current values given at a particular instant and classifies the data into no fault or the specific fault. The model is then interfaced with a real motor to accurately detect and classify the faults so that further necessary action can be taken.
       


### [Benchmarking Sim2Real Gap: High-fidelity Digital Twinning of Agile Manufacturing](https://arxiv.org/abs/2409.10784)

**Authors:**
Sunny Katyara, Suchita Sharma, Praveen Damacharla, Carlos Garcia Santiago, Lubina Dhirani, Bhawani Shankar Chowdhry

**Abstract:**
As the manufacturing industry shifts from mass production to mass customization, there is a growing emphasis on adopting agile, resilient, and human-centric methodologies in line with the directives of Industry 5.0. Central to this transformation is the deployment of digital twins, a technology that digitally replicates manufacturing assets to enable enhanced process optimization, predictive maintenance, synthetic data generation, and accelerated customization and prototyping. This chapter delves into the technologies underpinning the creation of digital twins specifically tailored to agile manufacturing scenarios within the realm of robotic automation. It explores the transfer of trained policies and process optimizations from simulated settings to real-world applications through advanced techniques such as domain randomization, domain adaptation, curriculum learning, and model-based system identification. The chapter also examines various industrial manufacturing automation scenarios, including bin-picking, part inspection, and product assembly, under Sim2Real conditions. The performance of digital twin technologies in these scenarios is evaluated using practical metrics including data latency, adaptation rate, simulation fidelity among others reported, providing a comprehensive assessment of their efficacy and potential impact on modern manufacturing processes.
       


### [Volvo Discovery Challenge at ECML-PKDD 2024](https://arxiv.org/abs/2409.11446)

**Authors:**
Mahmoud Rahat, Peyman Sheikholharam Mashhadi, Sławomir Nowaczyk, Shamik Choudhury, Leo Petrin, Thorsteinn Rognvaldsson, Andreas Voskou, Carlo Metta, Claudio Savelli

**Abstract:**
This paper presents an overview of the Volvo Discovery Challenge, held during the ECML-PKDD 2024 conference. The challenge's goal was to predict the failure risk of an anonymized component in Volvo trucks using a newly published dataset. The test data included observations from two generations (gen1 and gen2) of the component, while the training data was provided only for gen1. The challenge attracted 52 data scientists from around the world who submitted a total of 791 entries. We provide a brief description of the problem definition, challenge setup, and statistics about the submissions. In the section on winning methodologies, the first, second, and third-place winners of the competition briefly describe their proposed methods and provide GitHub links to their implemented code. The shared code can be interesting as an advanced methodology for researchers in the predictive maintenance domain. The competition was hosted on the Codabench platform.
       


### [Achieving Predictive Precision: Leveraging LSTM and Pseudo Labeling for Volvo's Discovery Challenge at ECML-PKDD 2024](https://arxiv.org/abs/2409.13877)

**Authors:**
Carlo Metta, Marco Gregnanin, Andrea Papini, Silvia Giulia Galfrè, Andrea Fois, Francesco Morandin, Marco Fantozzi, Maurizio Parton

**Abstract:**
This paper presents the second-place methodology in the Volvo Discovery Challenge at ECML-PKDD 2024, where we used Long Short-Term Memory networks and pseudo-labeling to predict maintenance needs for a component of Volvo trucks. We processed the training data to mirror the test set structure and applied a base LSTM model to label the test data iteratively. This approach refined our model's predictive capabilities and culminated in a macro-average F1-score of 0.879, demonstrating robust performance in predictive maintenance. This work provides valuable insights for applying machine learning techniques effectively in industrial settings.
       


### [Sparse Low-Ranked Self-Attention Transformer for Remaining Useful Lifetime Prediction of Optical Fiber Amplifiers](https://arxiv.org/abs/2409.14378)

**Authors:**
Dominic Schneider, Lutz Rapp

**Abstract:**
Optical fiber amplifiers are key elements in present optical networks. Failures of these components result in high financial loss of income of the network operator as the communication traffic over an affected link is interrupted. Applying Remaining useful lifetime (RUL) prediction in the context of Predictive Maintenance (PdM) to optical fiber amplifiers to predict upcoming system failures at an early stage, so that network outages can be minimized through planning of targeted maintenance actions, ensures reliability and safety. Optical fiber amplifier are complex systems, that work under various operating conditions, which makes correct forecasting a difficult task. Increased monitoring capabilities of systems results in datasets that facilitate the application of data-driven RUL prediction methods. Deep learning models in particular have shown good performance, but generalization based on comparatively small datasets for RUL prediction is difficult. In this paper, we propose Sparse Low-ranked self-Attention Transformer (SLAT) as a novel RUL prediction method. SLAT is based on an encoder-decoder architecture, wherein two parallel working encoders extract features for sensors and time steps. By utilizing the self-attention mechanism, long-term dependencies can be learned from long sequences. The implementation of sparsity in the attention matrix and a low-rank parametrization reduce overfitting and increase generalization. Experimental application to optical fiber amplifiers exemplified on EDFA, as well as a reference dataset from turbofan engines, shows that SLAT outperforms the state-of-the-art methods.
       


### [Domain knowledge-guided machine learning framework for state of health estimation in Lithium-ion batteries](https://arxiv.org/abs/2409.14575)

**Authors:**
Andrea Lanubile, Pietro Bosoni, Gabriele Pozzato, Anirudh Allam, Matteo Acquarone, Simona Onori

**Abstract:**
Accurate estimation of battery state of health is crucial for effective electric vehicle battery management. Here, we propose five health indicators that can be extracted online from real-world electric vehicle operation and develop a machine learning-based method to estimate the battery state of health. The proposed indicators provide physical insights into the energy and power fade of the battery and enable accurate capacity estimation even with partially missing data. Moreover, they can be computed for portions of the charging profile and real-world driving discharging conditions, facilitating real-time battery degradation estimation. The indicators are computed using experimental data from five cells aged under electric vehicle conditions, and a linear regression model is used to estimate the state of health. The results show that models trained with power autocorrelation and energy-based features achieve capacity estimation with maximum absolute percentage error within 1.5% to 2.5% .
       


### [XAI-guided Insulator Anomaly Detection for Imbalanced Datasets](https://arxiv.org/abs/2409.16821)

**Authors:**
Maximilian Andreas Hoefler, Karsten Mueller, Wojciech Samek

**Abstract:**
Power grids serve as a vital component in numerous industries, seamlessly delivering electrical energy to industrial processes and technologies, making their safe and reliable operation indispensable. However, powerlines can be hard to inspect due to difficult terrain or harsh climatic conditions. Therefore, unmanned aerial vehicles are increasingly deployed to inspect powerlines, resulting in a substantial stream of visual data which requires swift and accurate processing. Deep learning methods have become widely popular for this task, proving to be a valuable asset in fault detection. In particular, the detection of insulator defects is crucial for predicting powerline failures, since their malfunction can lead to transmission disruptions. It is therefore of great interest to continuously maintain and rigorously inspect insulator components. In this work we propose a novel pipeline to tackle this task. We utilize state-of-the-art object detection to detect and subsequently classify individual insulator anomalies. Our approach addresses dataset challenges such as imbalance and motion-blurred images through a fine-tuning methodology which allows us to alter the classification focus of the model by increasing the classification accuracy of anomalous insulators. In addition, we employ explainable-AI tools for precise localization and explanation of anomalies. This proposed method contributes to the field of anomaly detection, particularly vision-based industrial inspection and predictive maintenance. We significantly improve defect detection accuracy by up to 13%, while also offering a detailed analysis of model mis-classifications and localization quality, showcasing the potential of our method on real-world data.
       


### [Intelligent Energy Management: Remaining Useful Life Prediction and Charging Automation System Comprised of Deep Learning and the Internet of Things](https://arxiv.org/abs/2409.17931)

**Authors:**
Biplov Paneru, Bishwash Paneru, DP Sharma Mainali

**Abstract:**
Remaining Useful Life (RUL) of battery is an important parameter to know the battery's remaining life and need for recharge. The goal of this research project is to develop machine learning-based models for the battery RUL dataset. Different ML models are developed to classify the RUL of the vehicle, and the IoT (Internet of Things) concept is simulated for automating the charging system and managing any faults aligning. The graphs plotted depict the relationship between various vehicle parameters using the Blynk IoT platform. Results show that the catboost, Multi-Layer Perceptron (MLP), Gated Recurrent Unit (GRU), and hybrid model developed could classify RUL into three classes with 99% more accuracy. The data is fed using the tkinter GUI for simulating artificial intelligence (AI)-based charging, and with a pyserial backend, data can be entered into the Esp-32 microcontroller for making charge discharge possible with the model's predictions. Also, with an IoT system, the charging can be disconnected, monitored, and analyzed for automation. The results show that an accuracy of 99% can be obtained on models MLP, catboost model and similar accuracy on GRU model can be obtained, and finally relay-based triggering can be made by prediction through the model used for automating the charging and energy-saving mechanism. By showcasing an exemplary Blynk platform-based monitoring and automation phenomenon, we further present innovative ways of monitoring parameters and automating the system.
       


### [Canonical Correlation Guided Deep Neural Network](https://arxiv.org/abs/2409.19396)

**Authors:**
Zhiwen Chen, Siwen Mo, Haobin Ke, Steven X. Ding, Zhaohui Jiang, Chunhua Yang, Weihua Gui

**Abstract:**
Learning representations of two views of data such that the resulting representations are highly linearly correlated is appealing in machine learning. In this paper, we present a canonical correlation guided learning framework, which allows to be realized by deep neural networks (CCDNN), to learn such a correlated representation. It is also a novel merging of multivariate analysis (MVA) and machine learning, which can be viewed as transforming MVA into end-to-end architectures with the aid of neural networks. Unlike the linear canonical correlation analysis (CCA), kernel CCA and deep CCA, in the proposed method, the optimization formulation is not restricted to maximize correlation, instead we make canonical correlation as a constraint, which preserves the correlated representation learning ability and focuses more on the engineering tasks endowed by optimization formulation, such as reconstruction, classification and prediction. Furthermore, to reduce the redundancy induced by correlation, a redundancy filter is designed. We illustrate the performance of CCDNN on various tasks. In experiments on MNIST dataset, the results show that CCDNN has better reconstruction performance in terms of mean squared error and mean absolute error than DCCA and DCCAE. Also, we present the application of the proposed network to industrial fault diagnosis and remaining useful life cases for the classification and prediction tasks accordingly. The proposed method demonstrates superior performance in both tasks when compared to existing methods. Extension of CCDNN to much more deeper with the aid of residual connection is also presented in appendix.
       


### [A Survey on Graph Neural Networks for Remaining Useful Life Prediction: Methodologies, Evaluation and Future Trends](https://arxiv.org/abs/2409.19629)

**Authors:**
Yucheng Wang, Min Wu, Xiaoli Li, Lihua Xie, Zhenghua Chen

**Abstract:**
Remaining Useful Life (RUL) prediction is a critical aspect of Prognostics and Health Management (PHM), aimed at predicting the future state of a system to enable timely maintenance and prevent unexpected failures. While existing deep learning methods have shown promise, they often struggle to fully leverage the spatial information inherent in complex systems, limiting their effectiveness in RUL prediction. To address this challenge, recent research has explored the use of Graph Neural Networks (GNNs) to model spatial information for more accurate RUL prediction. This paper presents a comprehensive review of GNN techniques applied to RUL prediction, summarizing existing methods and offering guidance for future research. We first propose a novel taxonomy based on the stages of adapting GNNs to RUL prediction, systematically categorizing approaches into four key stages: graph construction, graph modeling, graph information processing, and graph readout. By organizing the field in this way, we highlight the unique challenges and considerations at each stage of the GNN pipeline. Additionally, we conduct a thorough evaluation of various state-of-the-art (SOTA) GNN methods, ensuring consistent experimental settings for fair comparisons. This rigorous analysis yields valuable insights into the strengths and weaknesses of different approaches, serving as an experimental guide for researchers and practitioners working in this area. Finally, we identify and discuss several promising research directions that could further advance the field, emphasizing the potential for GNNs to revolutionize RUL prediction and enhance the effectiveness of PHM strategies. The benchmarking codes are available in GitHub: https://github.com/Frank-Wang-oss/GNN\_RUL\_Benchmarking.
       


### [Online identification of skidding modes with interactive multiple model estimation](https://arxiv.org/abs/2409.20554)

**Authors:**
Ameya Salvi, Pardha Sai Krishna Ala, Jonathon M. Smereka, Mark Brudnak, David Gorsich, Matthias Schmid, Venkat Krovi

**Abstract:**
Skid-steered wheel mobile robots (SSWMRs) operate in a variety of outdoor environments exhibiting motion behaviors dominated by the effects of complex wheel-ground interactions. Characterizing these interactions is crucial both from the immediate robot autonomy perspective (for motion prediction and control) as well as a long-term predictive maintenance and diagnostics perspective. An ideal solution entails capturing precise state measurements for decisions and controls, which is considerably difficult, especially in increasingly unstructured outdoor regimes of operations for these robots. In this milieu, a framework to identify pre-determined discrete modes of operation can considerably simplify the motion model identification process. To this end, we propose an interactive multiple model (IMM) based filtering framework to probabilistically identify predefined robot operation modes that could arise due to traversal in different terrains or loss of wheel traction.
       


## October
### [Predicting Thermal Stress and Failure Risk in Monoblock Divertors Using 2D Finite Difference Modelling and Gradient Boosting Regression for Fusion Energy Applications](https://arxiv.org/abs/2410.02368)

**Author:**
Ayobami Daramola

**Abstract:**
This study presents a combined approach using a 2D finite difference method and Gradient Boosting Regressor (GBR) to analyze thermal stress and identify potential failure points in monoblock divertors made of tungsten, copper, and CuCrZr alloy. The model simulates temperature and heat flux distributions under typical fusion reactor conditions, highlighting regions of high thermal gradients and stress accumulation. These stress concentrations, particularly at the interfaces between materials, are key areas for potential failure, such as thermal fatigue and microcracking. Using the GBR model, a predictive maintenance framework is developed to assess failure risk based on thermal stress data, allowing for early intervention. This approach provides insights into the thermomechanical behavior of divertors, contributing to the design and maintenance of more resilient fusion reactor components.
       


### [Remaining Useful Life Prediction: A Study on Multidimensional Industrial Signal Processing and Efficient Transfer Learning Based on Large Language Models](https://arxiv.org/abs/2410.03134)

**Authors:**
Yan Chen, Cheng Liu

**Abstract:**
Remaining useful life (RUL) prediction is crucial for maintaining modern industrial systems, where equipment reliability and operational safety are paramount. Traditional methods, based on small-scale deep learning or physical/statistical models, often struggle with complex, multidimensional sensor data and varying operating conditions, limiting their generalization capabilities. To address these challenges, this paper introduces an innovative regression framework utilizing large language models (LLMs) for RUL prediction. By leveraging the modeling power of LLMs pre-trained on corpus data, the proposed model can effectively capture complex temporal dependencies and improve prediction accuracy. Extensive experiments on the Turbofan engine's RUL prediction task show that the proposed model surpasses state-of-the-art (SOTA) methods on the challenging FD002 and FD004 subsets and achieves near-SOTA results on the other subsets. Notably, different from previous research, our framework uses the same sliding window length and all sensor signals for all subsets, demonstrating strong consistency and generalization. Moreover, transfer learning experiments reveal that with minimal target domain data for fine-tuning, the model outperforms SOTA methods trained on full target domain data. This research highlights the significant potential of LLMs in industrial signal processing and RUL prediction, offering a forward-looking solution for health management in future intelligent industrial systems.
       


### [Enhanced Digital Twin for Human-Centric and Integrated Lighting Asset Management in Public Libraries: From Corrective to Predictive Maintenance](https://arxiv.org/abs/2410.03811)

**Authors:**
Jing Lin, Jingchun Shen

**Abstract:**
Lighting asset management in public libraries has traditionally been reactive, focusing on corrective maintenance, addressing issues only when failures occur. Although standards now encourage preventive measures, such as incorporating a maintenance factor, the broader goal of human centric, sustainable lighting systems requires a shift toward predictive maintenance strategies. This study introduces an enhanced digital twin model designed for the proactive management of lighting assets in public libraries. By integrating descriptive, diagnostic, predictive, and prescriptive analytics, the model enables a comprehensive, multilevel view of asset health. The proposed framework supports both preventive and predictive maintenance strategies, allowing for early detection of issues and the timely resolution of potential failures. In addition to the specific application for lighting systems, the design is adaptable for other building assets, providing a scalable solution for integrated asset management in various public spaces.
       


### [Understanding the irreversible lithium loss in silicon anodes using multi-edge X-ray scattering analysis](https://arxiv.org/abs/2410.05794)

**Authors:**
Michael A. Hernandez Bertran, Diana Zapata Dominguez, Christopher Berhaut, Samuel Tardif, Alessandro Longo, Christoph Sahle, Chiara Cavallari, Ivan Marri, Nathalie Herlin-Boime, Elisa Molinari, Stéphanie Pouget, Deborah Prezzi, Sandrine Lyonnard

**Abstract:**
During the first charge-discharge cycle, silicon-based batteries show an important capacity loss because of the formation of the solid electrolyte interphase (SEI) and morphological changes due to expansion-contraction sequence upon alloying. To understand this first-cycle irreversibility, quantitative methods are needed to characterize the chemical environment of silicon and lithium in the bulk of the cycled electrodes. Here we report a methodology based on multi-edge X-ray Raman Scattering performed on model silicon electrodes prepared in fully lithiated and fully delithiated states after the first cycle. The spectra were recorded at the C, O, F and Li K edges, as well as Si L2,3 edge. They were analysed using linear combinations of both experimental and computed reference spectra. We used prototypical SEI compounds as Li2CO3, LiF and LiPF6, as well as electrode constituents as binder and conductive carbon, cristalline Si, native SiO2,LixSi phases (x being the lithiation index) to identify the main species, isolate their relative contributions, and quantitatively evaluate the proportions of organic and inorganic products. We find that 30% of the carbonates formed in the SEI during the lithiation are dissolved on delithiation, and that part of the Li15Si4 alloys remain present after delithiation. By combining electrochemical analysis and XRS results, we identify that 17% of the lithium lost in the first cycle is trapped in disconnected silicon particles, while 30% form a fluorine-rich stable SEI and 53% a carbonate-rich partially-dissolvable SEI. These results pave the way to systematic, reference data-informed, and modelling assisted studies of SEI characteristics in the bulk of electrodes prepared under controlled state-of-charge and state-of-health conditions.
       


### [Predicting Battery Capacity Fade Using Probabilistic Machine Learning Models With and Without Pre-Trained Priors](https://arxiv.org/abs/2410.06422)

**Authors:**
Michael J. Kenney, Katerina G. Malollari, Sergei V. Kalinin, Maxim Ziatdinov

**Abstract:**
Lithium-ion batteries are a key energy storage technology driving revolutions in mobile electronics, electric vehicles and renewable energy storage. Capacity retention is a vital performance measure that is frequently utilized to assess whether these batteries have approached their end-of-life. Machine learning (ML) offers a powerful tool for predicting capacity degradation based on past data, and, potentially, prior physical knowledge, but the degree to which an ML prediction can be trusted is of significant practical importance in situations where consequential decisions must be made based on battery state of health. This study explores the efficacy of fully Bayesian machine learning in forecasting battery health with the quantification of uncertainty in its predictions. Specifically, we implemented three probabilistic ML approaches and evaluated the accuracy of their predictions and uncertainty estimates: a standard Gaussian process (GP), a structured Gaussian process (sGP), and a fully Bayesian neural network (BNN). In typical applications of GP and sGP, their hyperparameters are learned from a single sample while, in contrast, BNNs are typically pre-trained on an existing dataset to learn the weight distributions before being used for inference. This difference in methodology gives the BNN an advantage in learning global trends in a dataset and makes BNNs a good choice when training data is available. However, we show that pre-training can also be leveraged for GP and sGP approaches to learn the prior distributions of the hyperparameters and that in the case of the pre-trained sGP, similar accuracy and improved uncertainty estimation compared to the BNN can be achieved. This approach offers a framework for a broad range of probabilistic machine learning scenarios where past data is available and can be used to learn priors for (hyper)parameters of probabilistic ML models.
       


### [iFANnpp: Nuclear Power Plant Digital Twin for Robots and Autonomous Intelligence](https://arxiv.org/abs/2410.09213)

**Authors:**
Youndo Do, Marc Zebrowitz, Jackson Stahl, Fan Zhang

**Abstract:**
Robotics has gained significant attention due to its autonomy and ability to automate in the nuclear industry. However, the increasing complexity of robots has led to a growing demand for advanced simulation and control methods to predict robot behavior and optimize plant performance. Most existing digital twins only address parts of systems and do not offer an overall design of nuclear power plants. Furthermore, they are often designed for specific algorithms or tasks, making them unsuitable for broader research applications or other potential projects. In response, we propose a comprehensive nuclear power plant designed to enhance real-time monitoring, operational efficiency, and predictive maintenance. We selected to model a full-scope nuclear power plant in Unreal Engine 5 to incorporate the complexities and various phenomena. The high-resolution simulation environment is integrated with a General Pressurized Water Reactor Simulator, a high-fidelity physics-driven software, to create a realistic flow of nuclear power plant and a real-time updating virtual environment. Furthermore, the virtual environment provides various features and a Python bridge for researchers to test custom algorithms and frameworks easily. The digital twin's performance is presented, and several research ideas - such as multi-robot task scheduling and robot navigation in the radiation area - using implemented features are presented.
       


### [Uncertainty quantification on the prediction of creep remaining useful life](https://arxiv.org/abs/2410.10830)

**Authors:**
Victor Maudonet, Carlos Frederico Trotta Matt, Americo Cunha Jr

**Abstract:**
Accurate prediction of remaining useful life (RUL) under creep conditions is crucial for the design and maintenance of industrial equipment operating at high temperatures. Traditional deterministic methods often overlook significant variability in experimental data, leading to unreliable predictions. This study introduces a probabilistic framework to address uncertainties in predicting creep rupture time. We utilize robust regression methods to minimize the influence of outliers and enhance model estimates. Sobol indices-based global sensitivity analysis identifies the most influential parameters, followed by Monte Carlo simulations to determine the probability distribution of the material's RUL. Model selection techniques, including the Akaike and Bayesian information criteria, ensure the optimal predictive model. This probabilistic approach allows for the delineation of safe operational limits with quantifiable confidence levels, thereby improving the reliability and safety of high-temperature applications. The framework's versatility also allows integration with various mathematical models, offering a comprehensive understanding of creep behavior.
       


### [Evaluating Software Contribution Quality: Time-to-Modification Theory](https://arxiv.org/abs/2410.11768)

**Authors:**
Vincil Bishop III, Steven J Simske

**Abstract:**
The durability and quality of software contributions are critical factors in the long-term maintainability of a codebase. This paper introduces the Time to Modification (TTM) Theory, a novel approach for quantifying code quality by measuring the time interval between a code segment's introduction and its first modification. TTM serves as a proxy for code durability, with longer intervals suggesting higher-quality, more stable contributions. This work builds on previous research, including the "Time-Delta Method for Measuring Software Development Contribution Rates" dissertation, from which it heavily borrows concepts and methodologies. By leveraging version control systems such as Git, TTM provides granular insights into the temporal stability of code at various levels ranging from individual lines to entire repositories. TTM Theory contributes to the software engineering field by offering a dynamic metric that captures the evolution of a codebase over time, complementing traditional metrics like code churn and cyclomatic complexity. This metric is particularly useful for predicting maintenance needs, optimizing developer performance assessments, and improving the sustainability of software systems. Integrating TTM into continuous integration pipelines enables real-time monitoring of code stability, helping teams identify areas of instability and reduce technical debt.
       


### [Spatial-Temporal Bearing Fault Detection Using Graph Attention Networks and LSTM](https://arxiv.org/abs/2410.11923)

**Authors:**
Moirangthem Tiken Singh, Rabinder Kumar Prasad, Gurumayum Robert Michael, N. Hemarjit Singh, N. K. Kaphungkui

**Abstract:**
Purpose: This paper aims to enhance bearing fault diagnosis in industrial machinery by introducing a novel method that combines Graph Attention Network (GAT) and Long Short-Term Memory (LSTM) networks. This approach captures both spatial and temporal dependencies within sensor data, improving the accuracy of bearing fault detection under various conditions. Methodology: The proposed method converts time series sensor data into graph representations. GAT captures spatial relationships between components, while LSTM models temporal patterns. The model is validated using the Case Western Reserve University (CWRU) Bearing Dataset, which includes data under different horsepower levels and both normal and faulty conditions. Its performance is compared with methods such as K-Nearest Neighbors (KNN), Local Outlier Factor (LOF), Isolation Forest (IForest) and GNN-based method for bearing fault detection (GNNBFD). Findings: The model achieved outstanding results, with precision, recall, and F1-scores reaching 100\% across various testing conditions. It not only identifies faults accurately but also generalizes effectively across different operational scenarios, outperforming traditional methods. Originality: This research presents a unique combination of GAT and LSTM for fault detection, overcoming the limitations of traditional time series methods by capturing complex spatial-temporal dependencies. Its superior performance demonstrates significant potential for predictive maintenance in industrial applications.
       


### [Constrained Recurrent Bayesian Forecasting for Crack Propagation](https://arxiv.org/abs/2410.14761)

**Authors:**
Sara Yasmine Ouerk, Olivier Vo Van, Mouadh Yagoubi

**Abstract:**
Predictive maintenance of railway infrastructure, especially railroads, is essential to ensure safety. However, accurate prediction of crack evolution represents a major challenge due to the complex interactions between intrinsic and external factors, as well as measurement uncertainties. Effective modeling requires a multidimensional approach and a comprehensive understanding of these dynamics and uncertainties. Motivated by an industrial use case based on collected real data containing measured crack lengths, this paper introduces a robust Bayesian multi-horizon approach for predicting the temporal evolution of crack lengths on rails. This model captures the intricate interplay between various factors influencing crack growth. Additionally, the Bayesian approach quantifies both epistemic and aleatoric uncertainties, providing a confidence interval around predictions. To enhance the model's reliability for railroad maintenance, specific constraints are incorporated. These constraints limit non-physical crack propagation behavior and prioritize safety. The findings reveal a trade-off between prediction accuracy and constraint compliance, highlighting the nuanced decision-making process in model training. This study offers insights into advanced predictive modeling for dynamic temporal forecasting, particularly in railway maintenance, with potential applications in other domains.
       


### [Onboard Health Estimation using Distribution of Relaxation Times for Lithium-ion Batteries](https://arxiv.org/abs/2410.15271)

**Authors:**
Muhammad Aadil Khan, Sai Thatipamula, Simona Onori

**Abstract:**
Real-life batteries tend to experience a range of operating conditions, and undergo degradation due to a combination of both calendar and cycling aging. Onboard health estimation models typically use cycling aging data only, and account for at most one operating condition e.g., temperature, which can limit the accuracy of the models for state-of-health (SOH) estimation. In this paper, we utilize electrochemical impedance spectroscopy (EIS) data from 5 calendar-aged and 17 cycling-aged cells to perform SOH estimation under various operating conditions. The EIS curves are deconvoluted using the distribution of relaxation times (DRT) technique to map them onto a function $\textbf{g}$ which consists of distinct timescales representing different resistances inside the cell. These DRT curves, $\textbf{g}$, are then used as inputs to a long short-term memory (LSTM)-based neural network model for SOH estimation. We validate the model performance by testing it on ten different test sets, and achieve an average RMSPE of 1.69% across these sets.
       


### [Power Plays: Unleashing Machine Learning Magic in Smart Grids](https://arxiv.org/abs/2410.15423)

**Authors:**
Abdur Rashid, Parag Biswas, abdullah al masum, MD Abdullah Al Nasim, Kishor Datta Gupta

**Abstract:**
The integration of machine learning into smart grid systems represents a transformative step in enhancing the efficiency, reliability, and sustainability of modern energy networks. By adding advanced data analytics, these systems can better manage the complexities of renewable energy integration, demand response, and predictive maintenance. Machine learning algorithms analyze vast amounts of data from smart meters, sensors, and other grid components to optimize energy distribution, forecast demand, and detect irregularities that could indicate potential failures. This enables more precise load balancing, reduces operational costs, and enhances the resilience of the grid against disturbances. Furthermore, the use of predictive models helps in anticipating equipment failures, thereby improving the reliability of the energy supply. As smart grids continue to evolve, the role of machine learning in managing decentralized energy sources and enabling real-time decision-making will become increasingly critical. However, the deployment of these technologies also raises challenges related to data privacy, security, and the need for robust infrastructure. Addressing these issues in this research authors will focus on realizing the full potential of smart grids, ensuring they meet the growing energy demands while maintaining a focus on sustainability and efficiency using Machine Learning techniques. Furthermore, this research will help determine the smart grid's essentiality with the aid of Machine Learning. Multiple ML algorithms have been integrated along with their pros and cons. The future scope of these algorithms are also integrated.
       


### [Revisiting Deep Feature Reconstruction for Logical and Structural Industrial Anomaly Detection](https://arxiv.org/abs/2410.16255)

**Authors:**
Sukanya Patra, Souhaib Ben Taieb

**Abstract:**
Industrial anomaly detection is crucial for quality control and predictive maintenance, but it presents challenges due to limited training data, diverse anomaly types, and external factors that alter object appearances. Existing methods commonly detect structural anomalies, such as dents and scratches, by leveraging multi-scale features from image patches extracted through deep pre-trained networks. However, significant memory and computational demands often limit their practical application. Additionally, detecting logical anomalies-such as images with missing or excess elements-requires an understanding of spatial relationships that traditional patch-based methods fail to capture. In this work, we address these limitations by focusing on Deep Feature Reconstruction (DFR), a memory- and compute-efficient approach for detecting structural anomalies. We further enhance DFR into a unified framework, called ULSAD, which is capable of detecting both structural and logical anomalies. Specifically, we refine the DFR training objective to improve performance in structural anomaly detection, while introducing an attention-based loss mechanism using a global autoencoder-like network to handle logical anomaly detection. Our empirical evaluation across five benchmark datasets demonstrates the performance of ULSAD in detecting and localizing both structural and logical anomalies, outperforming eight state-of-the-art methods. An extensive ablation study further highlights the contribution of each component to the overall performance improvement. Our code is available at https://github.com/sukanyapatra1997/ULSAD-2024.git
       


### [Fast State-of-Health Estimation Method for Lithium-ion Battery using Sparse Identification of Nonlinear Dynamics](https://arxiv.org/abs/2410.16749)

**Authors:**
Jayden Dongwoo Lee, Donghoon Seo, Jongho Shin, Hyochoong Bang

**Abstract:**
Lithium-ion batteries (LIBs) are utilized as a major energy source in various fields because of their high energy density and long lifespan. During repeated charging and discharging, the degradation of LIBs, which reduces their maximum power output and operating time, is a pivotal issue. This degradation can affect not only battery performance but also safety of the system. Therefore, it is essential to accurately estimate the state-of-health (SOH) of the battery in real time. To address this problem, we propose a fast SOH estimation method that utilizes the sparse model identification algorithm (SINDy) for nonlinear dynamics. SINDy can discover the governing equations of target systems with low data assuming that few functions have the dominant characteristic of the system. To decide the state of degradation model, correlation analysis is suggested. Using SINDy and correlation analysis, we can obtain the data-driven SOH model to improve the interpretability of the system. To validate the feasibility of the proposed method, the estimation performance of the SOH and the computation time are evaluated by comparing it with various machine learning algorithms.
       


### [Electrode SOC and SOH estimation with electrode-level ECMs](https://arxiv.org/abs/2410.16980)

**Authors:**
Iker Lopetegi, Sergio Fernandez, Gregory L. Plett, M. Scott Trimboli, Unai Iraola

**Abstract:**
Being able to predict battery internal states that are related to battery degradation is a key aspect to improve battery lifetime and performance, enhancing cleaner electric transportation and energy generation. However, most present battery management systems (BMSs) use equivalent-circuit models (ECMs) for state of charge (SOC) and state of health (SOH) estimation. These models are not able to predict these aging-related variables, and therefore, they cannot be used to limit battery degradation. In this paper, we propose a method for electrode-level SOC (eSOC) and electrode-level SOH (eSOH) estimation using an electrode-level ECM (eECM). The method can produce estimates of the states of lithiation (SOL) of both electrodes and update the eSOH parameters to maintain estimation accuracy through the lifetime of the battery. Furthermore, the eSOH parameter estimates are used to obtain degradation mode information, which could be used to improve state estimation, health diagnosis and prognosis. The method was validated in simulation and experimentally.
       


### [Smart Transport Infrastructure Maintenance: A Smart-Contract Blockchain Approach](https://arxiv.org/abs/2410.20431)

**Author:**
Fatjon Seraj

**Abstract:**
Infrastructure maintenance is inherently complex, especially for widely dispersed transport systems like roads and railroads. Maintaining this infrastructure involves multiple partners working together to ensure safe, efficient upkeep that meets technical and safety standards, with timely materials and budget adherence. Traditionally, these requirements are managed on paper, with each contract step checked manually. Smart contracts, based on blockchain distributed ledger technology, offer a new approach. Distributed ledgers facilitate secure, transparent transactions, enabling decentralized agreements where contract terms automatically execute when conditions are met. Beyond financial transactions, blockchains can track complex agreements, recording each stage of contract fulfillment between multiple parties. A smart contract is a set of coded rules stored on the blockchain that automatically executes each term upon meeting specified conditions. In infrastructure maintenance, this enables end-to-end automation-from contractor assignment to maintenance completion. Using an immutable, decentralized record, contract terms and statuses are transparent to all parties, enhancing trust and efficiency. Creating smart contracts for infrastructure requires a comprehensive understanding of procedural workflows to foresee all requirements and liabilities. This workflow includes continuous infrastructure monitoring through a dynamic, data-driven maintenance model that triggers necessary actions. Modern process mining can develop a resilient Maintenance Process Model, helping Operations Management to define contract terms, including asset allocation, logistics, materials, and skill requirements. Automation and reliable data quality across the procedural chain are essential, supported by IoT sensors, big data analytics, predictive maintenance, intelligent logistics, and asset management.
       


### [A Robust Topological Framework for Detecting Regime Changes in Multi-Trial Experiments with Application to Predictive Maintenance](https://arxiv.org/abs/2410.20443)

**Authors:**
Anass B. El-Yaagoubi, Jean-Marc Freyermuth, Hernando Ombao

**Abstract:**
We present a general and flexible framework for detecting regime changes in complex, non-stationary data across multi-trial experiments. Traditional change point detection methods focus on identifying abrupt changes within a single time series (single trial), targeting shifts in statistical properties such as the mean, variance, and spectrum over time within that sole trial. In contrast, our approach considers changes occurring across trials, accommodating changes that may arise within individual trials due to experimental inconsistencies, such as varying delays or event duration. By leveraging diverse metrics to analyze time-frequency characteristics specifically topological changes in the spectrum and spectrograms, our approach offers a comprehensive framework for detecting such variations. Our approach can handle different scenarios with various statistical assumptions, including varying levels of stationarity within and across trials, making our framework highly adaptable. We validate our approach through simulations using time-varying autoregressive processes that exhibit different regime changes. Our results demonstrate the effectiveness of detecting changes across trials under diverse conditions. Furthermore, we illustrate the effectiveness of our method by applying it to predictive maintenance using the NASA bearing dataset. By analyzing the time-frequency characteristics of vibration signals recorded by accelerometers, our approach accurately identifies bearing failures, showcasing its strong potential for early fault detection in mechanical systems.
       


### [Gaussian Derivative Change-point Detection for Early Warnings of Industrial System Failures](https://arxiv.org/abs/2410.22594)

**Authors:**
Hao Zhao, Rong Pan

**Abstract:**
An early warning of future system failure is essential for conducting predictive maintenance and enhancing system availability. This paper introduces a three-step framework for assessing system health to predict imminent system breakdowns. First, the Gaussian Derivative Change-Point Detection (GDCPD) algorithm is proposed for detecting changes in the high-dimensional feature space. GDCPD conducts a multivariate Change-Point Detection (CPD) by implementing Gaussian derivative processes for identifying change locations on critical system features, as these changes eventually will lead to system failure. To assess the significance of these changes, Weighted Mahalanobis Distance (WMD) is applied in both offline and online analyses. In the offline setting, WMD helps establish a threshold that determines significant system variations, while in the online setting, it facilitates real-time monitoring, issuing alarms for potential future system breakdowns. Utilizing the insights gained from the GDCPD and monitoring scheme, Long Short-Term Memory (LSTM) network is then employed to estimate the Remaining Useful Life (RUL) of the system. The experimental study of a real-world system demonstrates the effectiveness of the proposed methodology in accurately forecasting system failures well before they occur. By integrating CPD with real-time monitoring and RUL prediction, this methodology significantly advances system health monitoring and early warning capabilities.
       


### [DiffBatt: A Diffusion Model for Battery Degradation Prediction and Synthesis](https://arxiv.org/abs/2410.23893)

**Authors:**
Hamidreza Eivazi, André Hebenbrock, Raphael Ginster, Steffen Blömeke, Stefan Wittek, Christoph Herrmann, Thomas S. Spengler, Thomas Turek, Andreas Rausch

**Abstract:**
Battery degradation remains a critical challenge in the pursuit of green technologies and sustainable energy solutions. Despite significant research efforts, predicting battery capacity loss accurately remains a formidable task due to its complex nature, influenced by both aging and cycling behaviors. To address this challenge, we introduce a novel general-purpose model for battery degradation prediction and synthesis, DiffBatt. Leveraging an innovative combination of conditional and unconditional diffusion models with classifier-free guidance and transformer architecture, DiffBatt achieves high expressivity and scalability. DiffBatt operates as a probabilistic model to capture uncertainty in aging behaviors and a generative model to simulate battery degradation. The performance of the model excels in prediction tasks while also enabling the generation of synthetic degradation curves, facilitating enhanced model training by data augmentation. In the remaining useful life prediction task, DiffBatt provides accurate results with a mean RMSE of 196 cycles across all datasets, outperforming all other models and demonstrating superior generalizability. This work represents an important step towards developing foundational models for battery degradation.
       


## November
### [SambaMixer: State of Health Prediction of Li-ion Batteries using Mamba State Space Models](https://arxiv.org/abs/2411.00233)

**Authors:**
José Ignacio Olalde-Verano, Sascha Kirch, Clara Pérez-Molina, Sergio Martin

**Abstract:**
The state of health (SOH) of a Li-ion battery is a critical parameter that determines the remaining capacity and the remaining lifetime of the battery. In this paper, we propose SambaMixer a novel structured state space model (SSM) for predicting the state of health of Li-ion batteries. The proposed SSM is based on the MambaMixer architecture, which is designed to handle multi-variate time signals. We evaluate our model on the NASA battery discharge dataset and show that our model outperforms the state-of-the-art on this dataset. We further introduce a novel anchor-based resampling method which ensures time signals are of the expected length while also serving as augmentation technique. Finally, we condition prediction on the sample time and the cycle time difference using positional encodings to improve the performance of our model and to learn recuperation effects. Our results proof that our model is able to predict the SOH of Li-ion batteries with high accuracy and robustness.
       


### [A Multi-Granularity Supervised Contrastive Framework for Remaining Useful Life Prediction of Aero-engines](https://arxiv.org/abs/2411.00461)

**Authors:**
Zixuan He, Ziqian Kong, Zhengyu Chen, Yuling Zhan, Zijun Que, Zhengguo Xu

**Abstract:**
Accurate remaining useful life (RUL) predictions are critical to the safe operation of aero-engines. Currently, the RUL prediction task is mainly a regression paradigm with only mean square error as the loss function and lacks research on feature space structure, the latter of which has shown excellent performance in a large number of studies. This paper develops a multi-granularity supervised contrastive (MGSC) framework from plain intuition that samples with the same RUL label should be aligned in the feature space, and address the problems of too large minibatch size and unbalanced samples in the implementation. The RUL prediction with MGSC is implemented on using the proposed multi-phase training strategy. This paper also demonstrates a simple and scalable basic network structure and validates the proposed MGSC strategy on the CMPASS dataset using a convolutional long short-term memory network as a baseline, which effectively improves the accuracy of RUL prediction.
       


### [PMI-DT: Leveraging Digital Twins and Machine Learning for Predictive Modeling and Inspection in Manufacturing](https://arxiv.org/abs/2411.01299)

**Authors:**
Chas Hamel, Md Manjurul Ahsan, Shivakumar Raman

**Abstract:**
Over the years, Digital Twin (DT) has become popular in Advanced Manufacturing (AM) due to its ability to improve production efficiency and quality. By creating virtual replicas of physical assets, DTs help in real-time monitoring, develop predictive models, and improve operational performance. However, integrating data from physical systems into reliable predictive models, particularly in precision measurement and failure prevention, is often challenging and less explored. This study introduces a Predictive Maintenance and Inspection Digital Twin (PMI-DT) framework with a focus on precision measurement and predictive quality assurance using 3D-printed 1''-4 ACME bolt, CyberGage 360 vision inspection system, SolidWorks, and Microsoft Azure. During this approach, dimensional inspection data is combined with fatigue test results to create a model for detecting failures. Using Machine Learning (ML) -- Random Forest and Decision Tree models -- the proposed approaches were able to predict bolt failure with real-time data 100% accurately. Our preliminary result shows Max Position (30%) and Max Load (24%) are the main factors that contribute to that failure. We expect the PMI-DT framework will reduce inspection time and improve predictive maintenance, ultimately giving manufacturers a practical way to boost product quality and reliability using DT in AM.
       


### [TwiNet: Connecting Real World Networks to their Digital Twins Through a Live Bidirectional Link](https://arxiv.org/abs/2411.03503)

**Authors:**
Clifton Paul Robinson, Andrea Lacava, Pedram Johari, Francesca Cuomo, Tommaso Melodia

**Abstract:**
The wireless spectrum's increasing complexity poses challenges and opportunities, highlighting the necessity for real-time solutions and robust data processing capabilities. Digital Twin (DT), virtual replicas of physical systems, integrate real-time data to mirror their real-world counterparts, enabling precise monitoring and optimization. Incorporating DTs into wireless communication enhances predictive maintenance, resource allocation, and troubleshooting, thus bolstering network reliability. Our paper introduces TwiNet, enabling bidirectional, near-realtime links between real-world wireless spectrum scenarios and DT replicas. Utilizing the protocol, MQTT, we can achieve data transfer times with an average latency of 14 ms, suitable for real-time communication. This is confirmed by monitoring real-world traffic and mirroring it in real-time within the DT's wireless environment. We evaluate TwiNet's performance in two use cases: (i) assessing risky traffic configurations of UEs in a Safe Adaptive Data Rate (SADR) system, improving network performance by approximately 15% compared to original network selections; and (ii) deploying new CNNs in response to jammed pilots, achieving up to 97% accuracy training on artificial data and deploying a new model in as low as 2 minutes to counter persistent adversaries. TwiNet enables swift deployment and adaptation of DTs, addressing crucial challenges in modern wireless communication systems.
       


### [A Practical Example of the Impact of Uncertainty on the One-Dimensional Single-Diode Model](https://arxiv.org/abs/2411.04768)

**Authors:**
Carlos Cárdenas-Bravo, Sylvain Lespinats, Denys Dutykh

**Abstract:**
The state of health of solar photovoltaic (PV) systems is assessed by measuring the current-voltage (I-V) curves, which present a collection of three cardinal points: the short-circuit point, the open-circuit point, and the maximum power point. To understand the response of PV systems, the I-V curve is typically modeled using the well-known single-diode model (SDM), which involves five parameters. However, the SDM can be expressed as a function of one parameter when the information of the cardinal points is incorporated into the formulation. This paper presents a methodology to address the uncertainty of the cardinal points on the parameters of the single-diode model based on the mathematical theory. Utilizing the one-dimensional single-diode model as the basis, the study demonstrates that it is possible to include the uncertainty by solving a set of nonlinear equations. The results highlight the feasibility and effectiveness of this approach in accounting for uncertainties in the SDM parameters.
       


### [Metrology and Manufacturing-Integrated Digital Twin (MM-DT) for Advanced Manufacturing: Insights from CMM and FARO Arm Measurements](https://arxiv.org/abs/2411.05286)

**Authors:**
Hamidreza Samadi, Md Manjurul Ahsan, Shivakumar Raman

**Abstract:**
Metrology, the science of measurement, plays a key role in Advanced Manufacturing (AM) to ensure quality control, process optimization, and predictive maintenance. However, it has often been overlooked in AM domains due to the current focus on automation and the complexity of integrated precise measurement systems. Over the years, Digital Twin (DT) technology in AM has gained much attention due to its potential to address these challenges through physical data integration and real-time monitoring, though its use in metrology remains limited. Taking this into account, this study proposes a novel framework, the Metrology and Manufacturing-Integrated Digital Twin (MM-DT), which focuses on data from two metrology tools, collected from Coordinate Measuring Machines (CMM) and FARO Arm devices. Throughout this process, we measured 20 manufacturing parts, with each part assessed twice under different temperature conditions. Using Ensemble Machine Learning methods, our proposed approach predicts measurement deviations accurately, achieving an R2 score of 0.91 and reducing the Root Mean Square Error (RMSE) to 1.59 micrometers. Our MM-DT framework demonstrates its efficiency by improving metrology processes and offers valuable insights for researchers and practitioners who aim to increase manufacturing precision and quality.
       


### [Cell Balancing Paradigms: Advanced Types, Algorithms, and Optimization Frameworks](https://arxiv.org/abs/2411.05478)

**Authors:**
Anupama R Itagi, Rakhee Kallimani, Krishna Pai, Sridhar Iyer, Onel L. A. López, Sushant Mutagekar

**Abstract:**
The operation efficiency of the electric transportation, energy storage, and grids mainly depends on the fundamental characteristics of the employed batteries. Fundamental variables like voltage, current, temperature, and estimated parameters, like the State of Charge (SoC) of the battery pack, influence the functionality of the system. This motivates the implementation of a Battery Management System (BMS), critical for managing and maintaining the health, safety, and performance of a battery pack. This is ensured by measuring parameters like temperature, cell voltage, and pack current. It also involves monitoring insulation levels and fire hazards, while assessing the prevailing useful life of the batteries and estimating the SoC and State of Health (SoH). Additionally, the system manages and controls key activities like cell balancing and charge/discharge processes. Thus functioning of the battery can be optimised, by guaranteeing the vital parameters to be well within the prescribed range. This article discusses the several cell balancing schemes, and focuses on the intricacies of cell balancing algorithms and optimisation methods for cell balancing. We begin surveying recent cell balancing algorithms and then provide selection guidelines taking into account their advantages, disadvantages, and applications. Finally, we discuss various optimization algorithms and outline the essential parameters involved in the cell balancing process.
       


### [Fast Stochastic Subspace Identification of Densely Instrumented Bridges Using Randomized SVD](https://arxiv.org/abs/2411.05510)

**Authors:**
Elisa Tomassini, Enrique García-Macías, Filippo Ubertini

**Abstract:**
The rising number of bridge collapses worldwide has compelled governments to introduce predictive maintenance strategies to extend structural lifespan. In this context, vibration-based Structural Health Monitoring (SHM) techniques utilizing Operational Modal Analysis (OMA) are favored for their non-destructive and global assessment capabilities. However, long multi-span bridges instrumented with dense arrays of accelerometers present a particular challenge, as the computational demands of classical OMA techniques in such cases are incompatible with long-term SHM. To address this issue, this paper introduces Randomized Singular Value Decomposition (RSVD) as an efficient alternative to traditional SVD within Covariance-driven Stochastic Subspace Identification (CoV-SSI). The efficacy of RSVD is also leveraged to enhance modal identification results and reduce the need for expert intervention by means of 3D stabilization diagrams, which facilitate the investigation of the modal estimates over different model orders and time lags. The approach's effectiveness is demonstrated on the San Faustino Bridge in Italy, equipped with over 60 multiaxial accelerometers.
       


### [Multivariate Data Augmentation for Predictive Maintenance using Diffusion](https://arxiv.org/abs/2411.05848)

**Authors:**
Andrew Thompson, Alexander Sommers, Alicia Russell-Gilbert, Logan Cummins, Sudip Mittal, Shahram Rahimi, Maria Seale, Joseph Jaboure, Thomas Arnold, Joshua Church

**Abstract:**
Predictive maintenance has been used to optimize system repairs in the industrial, medical, and financial domains. This technique relies on the consistent ability to detect and predict anomalies in critical systems. AI models have been trained to detect system faults, improving predictive maintenance efficiency. Typically there is a lack of fault data to train these models, due to organizations working to keep fault occurrences and down time to a minimum. For newly installed systems, no fault data exists since they have yet to fail. By using diffusion models for synthetic data generation, the complex training datasets for these predictive models can be supplemented with high level synthetic fault data to improve their performance in anomaly detection. By learning the relationship between healthy and faulty data in similar systems, a diffusion model can attempt to apply that relationship to healthy data of a newly installed system that has no fault data. The diffusion model would then be able to generate useful fault data for the new system, and enable predictive models to be trained for predictive maintenance. The following paper demonstrates a system for generating useful, multivariate synthetic data for predictive maintenance, and how it can be applied to systems that have yet to fail.
       


### [Enhancing Predictive Maintenance in Mining Mobile Machinery through a TinyML-enabled Hierarchical Inference Network](https://arxiv.org/abs/2411.07168)

**Authors:**
Raúl de la Fuente, Luciano Radrigan, Anibal S Morales

**Abstract:**
Mining machinery operating in variable environments faces high wear and unpredictable stress, challenging Predictive Maintenance (PdM). This paper introduces the Edge Sensor Network for Predictive Maintenance (ESN-PdM), a hierarchical inference framework across edge devices, gateways, and cloud services for real-time condition monitoring. The system dynamically adjusts inference locations--on-device, on-gateway, or on-cloud--based on trade-offs among accuracy, latency, and battery life, leveraging Tiny Machine Learning (TinyML) techniques for model optimization on resource-constrained devices. Performance evaluations showed that on-sensor and on-gateway inference modes achieved over 90\% classification accuracy, while cloud-based inference reached 99\%. On-sensor inference reduced power consumption by approximately 44\%, enabling up to 104 hours of operation. Latency was lowest for on-device inference (3.33 ms), increasing when offloading to the gateway (146.67 ms) or cloud (641.71 ms). The ESN-PdM framework provides a scalable, adaptive solution for reliable anomaly detection and PdM, crucial for maintaining machinery uptime in remote environments. By balancing accuracy, latency, and energy consumption, this approach advances PdM frameworks for industrial applications.
       


### [A Fuzzy Reinforcement LSTM-based Long-term Prediction Model for Fault Conditions in Nuclear Power Plants](https://arxiv.org/abs/2411.08370)

**Authors:**
Siwei Li, Jiayan Fang, Yichun Wua, Wei Wang, Chengxin Li, Jiangwen Chen

**Abstract:**
Early fault detection and timely maintenance scheduling can significantly mitigate operational risks in NPPs and enhance the reliability of operator decision-making. Therefore, it is necessary to develop an efficient Prognostics and Health Management (PHM) multi-step prediction model for predicting of system health status and prompt execution of maintenance operations. In this study, we propose a novel predictive model that integrates reinforcement learning with Long Short-Term Memory (LSTM) neural networks and the Expert Fuzzy Evaluation Method. The model is validated using parameter data for 20 different breach sizes in the Main Steam Line Break (MSLB) accident condition of the CPR1000 pressurized water reactor simulation model and it demonstrates a remarkable capability in accurately forecasting NPP parameter changes up to 128 steps ahead (with a time interval of 10 seconds per step, i.e., 1280 seconds), thereby satisfying the temporal advance requirement for fault prognostics in NPPs. Furthermore, this method provides an effective reference solution for PHM applications such as anomaly detection and remaining useful life prediction.
       


### [Recommender systems and reinforcement learning for human-building interaction and context-aware support: A text mining-driven review of scientific literature](https://arxiv.org/abs/2411.08734)

**Authors:**
Wenhao Zhang, Matias Quintana, Clayton Miller

**Abstract:**
The indoor environment significantly impacts human health and well-being; enhancing health and reducing energy consumption in these settings is a central research focus. With the advancement of Information and Communication Technology (ICT), recommendation systems and reinforcement learning (RL) have emerged as promising approaches to induce behavioral changes to improve the indoor environment and energy efficiency of buildings. This study aims to employ text mining and Natural Language Processing (NLP) techniques to thoroughly examine the connections among these approaches in the context of human-building interaction and occupant context-aware support. The study analyzed 27,595 articles from the ScienceDirect database, revealing extensive use of recommendation systems and RL for space optimization, location recommendations, and personalized control suggestions. Although these systems are broadly applied to specific content, their use in optimizing indoor environments and energy efficiency remains limited. This gap likely arises from the need for interdisciplinary knowledge and extensive sensor data. Traditional recommendation algorithms, including collaborative filtering, content-based and knowledge-based methods, are commonly employed. However, the more complex challenges of optimizing indoor conditions and energy efficiency often depend on sophisticated machine learning (ML) techniques like reinforcement and deep learning. Furthermore, this review underscores the vast potential for expanding recommender systems and RL applications in buildings and indoor environments. Fields ripe for innovation include predictive maintenance, building-related product recommendation, and optimization of environments tailored for specific needs, such as sleep and productivity enhancements based on user feedback.
       


### [System Reliability Engineering in the Age of Industry 4.0: Challenges and Innovations](https://arxiv.org/abs/2411.08913)

**Authors:**
Antoine Tordeux, Tim M. Julitz, Isabelle Müller, Zikai Zhang, Jannis Pietruschka, Nicola Fricke, Nadine Schlüter, Stefan Bracke, Manuel Löwer

**Abstract:**
In the era of Industry 4.0, system reliability engineering faces both challenges and opportunities. On the one hand, the complexity of cyber-physical systems, the integration of novel numerical technologies, and the handling of large amounts of data pose new difficulties for ensuring system reliability. On the other hand, innovations such as AI-driven prognostics, digital twins, and IoT-enabled systems enable the implementation of new methodologies that are transforming reliability engineering. Condition-based monitoring and predictive maintenance are examples of key advancements, leveraging real-time sensor data collection and AI to predict and prevent equipment failures. These approaches reduce failures and downtime, lower costs, and extend equipment lifespan and sustainability. However, it also brings challenges such as data management, integrating complexity, and the need for fast and accurate models and algorithms. Overall, the convergence of advanced technologies in Industry 4.0 requires a rethinking of reliability tasks, emphasising adaptability and real-time data processing. In this chapter, we propose to review recent innovations in the field, related methods and applications, as well as challenges and barriers that remain to be explored. In the red lane, we focus on smart manufacturing and automotive engineering applications with sensor-based monitoring and driver assistance systems.
       


### [Sensor-fusion based Prognostics Framework for Complex Engineering Systems Exhibiting Multiple Failure Modes](https://arxiv.org/abs/2411.12159)

**Authors:**
Benjamin Peters, Ayush Mohanty, Xiaolei Fang, Stephen K. Robinson, Nagi Gebraeel

**Abstract:**
Complex engineering systems are often subject to multiple failure modes. Developing a remaining useful life (RUL) prediction model that does not consider the failure mode causing degradation is likely to result in inaccurate predictions. However, distinguishing between causes of failure without manually inspecting the system is nontrivial. This challenge is increased when the causes of historically observed failures are unknown. Sensors, which are useful for monitoring the state-of-health of systems, can also be used for distinguishing between multiple failure modes as the presence of multiple failure modes results in discriminatory behavior of the sensor signals. When systems are equipped with multiple sensors, some sensors may exhibit behavior correlated with degradation, while other sensors do not. Furthermore, which sensors exhibit this behavior may differ for each failure mode. In this paper, we present a simultaneous clustering and sensor selection approach for unlabeled training datasets of systems exhibiting multiple failure modes. The cluster assignments and the selected sensors are then utilized in real-time to first diagnose the active failure mode and then to predict the system RUL. We validate the complete pipeline of the methodology using a simulated dataset of systems exhibiting two failure modes and on a turbofan degradation dataset from NASA.
       


### [Gas-induced bulging in pouch-cell batteries: a mechanical model](https://arxiv.org/abs/2411.13197)

**Authors:**
Andrea Giudici, Colin Please, Jon Chapman

**Abstract:**
Over the long timescale of many charge/discharge cycles, gas formation can result in large bulging deformations of a Lithium-ion pouch cell, which is a key failure mechanism in batteries. Guided by recent experimental X-ray tomography data of a bulging cell, we propose a homogenised mechanical model to predict the shape of the deformation and the stress distribution analytically. Our model can be included in battery simulation models to capture the effects of mechanical degradation. Furthermore, with knowledge of the bending stiffness of the cathode electrodes and current collectors, and by fitting our model to experimental data, we can predict the internal pressure and the amount of gas in the battery, thus assisting in monitoring the state of health (SOH) of the cell without breaking the sealed case.
       


### [Executable QR codes with Machine Learning for Industrial Applications](https://arxiv.org/abs/2411.13400)

**Authors:**
Stefano Scanzio, Francesco Velluto, Matteo Rosani, Lukasz Wisniewski, Gianluca Cena

**Abstract:**
Executable QR codes, also known as eQR codes or just sQRy, are a special kind of QR codes that embed programs conceived to run on mobile devices like smartphones. Since the program is directly encoded in binary form within the QR code, it can be executed even when the reading device is not provided with Internet access. The applications of this technology are manifold, and range from smart user guides to advisory systems. The first programming language made available for eQR is QRtree, which enables the implementation of decision trees aimed, for example, at guiding the user in operating/maintaining a complex machinery or for reaching a specific location.
  In this work, an additional language is proposed, we term QRind, which was specifically devised for Industry. It permits to integrate distinct computational blocks into the QR code, e.g., machine learning models to enable predictive maintenance and algorithms to ease machinery usage. QRind permits the Industry 4.0/5.0 paradigms to be implemented, in part, also in those cases where Internet is unavailable.
       


### [Predictive Maintenance Study for High-Pressure Industrial Compressors: Hybrid Clustering Models](https://arxiv.org/abs/2411.13919)

**Authors:**
Alessandro Costa, Emilio Mastriani, Federico Incardona, Kevin Munari, Sebastiano Spinello

**Abstract:**
This study introduces a predictive maintenance strategy for high pressure industrial compressors using sensor data and features derived from unsupervised clustering integrated into classification models. The goal is to enhance model accuracy and efficiency in detecting compressor failures. After data pre processing, sensitive clustering parameters were tuned to identify algorithms that best capture the dataset's temporal and operational characteristics. Clustering algorithms were evaluated using quality metrics like Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI), selecting those most effective at distinguishing between normal and non normal conditions. These features enriched regression models, improving failure detection accuracy by 4.87 percent on average. Although training time was reduced by 22.96 percent, the decrease was not statistically significant, varying across algorithms. Cross validation and key performance metrics confirmed the benefits of clustering based features in predictive maintenance models.
       


### [Industrial Machines Health Prognosis using a Transformer-based Framework](https://arxiv.org/abs/2411.14443)

**Authors:**
David J Poland, Lemuel Puglisi, Daniele Ravi

**Abstract:**
This article introduces Transformer Quantile Regression Neural Networks (TQRNNs), a novel data-driven solution for real-time machine failure prediction in manufacturing contexts. Our objective is to develop an advanced predictive maintenance model capable of accurately identifying machine system breakdowns. To do so, TQRNNs employ a two-step approach: (i) a modified quantile regression neural network to segment anomaly outliers while maintaining low time complexity, and (ii) a concatenated transformer network aimed at facilitating accurate classification even within a large timeframe of up to one hour. We have implemented our proposed pipeline in a real-world beverage manufacturing industry setting. Our findings demonstrate the model's effectiveness, achieving an accuracy rate of 70.84% with a 1-hour lead time for predicting machine breakdowns. Additionally, our analysis shows that using TQRNNs can increase high-quality production, improving product yield from 78.38% to 89.62%. We believe that predictive maintenance assumes a pivotal role in modern manufacturing, minimizing unplanned downtime, reducing repair costs, optimizing production efficiency, and ensuring operational stability. Its potential to generate substantial cost savings while enhancing sustainability and competitiveness underscores its importance in contemporary manufacturing practices.
       


### [Many happy returns: machine learning to support platelet issuing and waste reduction in hospital blood banks](https://arxiv.org/abs/2411.14939)

**Authors:**
Joseph Farrington, Samah Alimam, Martin Utley, Kezhi Li, Wai Keong Wong

**Abstract:**
Efforts to reduce platelet wastage in hospital blood banks have focused on ordering policies, but the predominant practice of issuing the oldest unit first may not be optimal when some units are returned unused. We propose a novel, machine learning (ML)-guided issuing policy to increase the likelihood of returned units being reissued before expiration. Our ML model trained to predict returns on 17,297 requests for platelets gave AUROC 0.74 on 9,353 held-out requests. Prior to ML model development we built a simulation of the blood bank operation that incorporated returns to understand the scale of benefits of such a model. Using our trained model in the simulation gave an estimated reduction in wastage of 14%. Our partner hospital is considering adopting our approach, which would be particularly beneficial for hospitals with higher return rates and where units have a shorter remaining useful life on arrival.
       


### [Hybrid Gaussian Process Regression with Temporal Feature Extraction for Partially Interpretable Remaining Useful Life Interval Prediction in Aeroengine Prognostics](https://arxiv.org/abs/2411.15185)

**Authors:**
Tian Niu, Zijun Xu, Heng Luo, Ziqing Zhou

**Abstract:**
The estimation of Remaining Useful Life (RUL) plays a pivotal role in intelligent manufacturing systems and Industry 4.0 technologies. While recent advancements have improved RUL prediction, many models still face interpretability and compelling uncertainty modeling challenges. This paper introduces a modified Gaussian Process Regression (GPR) model for RUL interval prediction, tailored for the complexities of manufacturing process development. The modified GPR predicts confidence intervals by learning from historical data and addresses uncertainty modeling in a more structured way. The approach effectively captures intricate time-series patterns and dynamic behaviors inherent in modern manufacturing systems by coupling GPR with deep adaptive learning-enhanced AI process models. Moreover, the model evaluates feature significance to ensure more transparent decision-making, which is crucial for optimizing manufacturing processes. This comprehensive approach supports more accurate RUL predictions and provides transparent, interpretable insights into uncertainty, contributing to robust process development and management.
       


### [Turbofan Engine Remaining Useful Life (RUL) Prediction Based on Bi-Directional Long Short-Term Memory (BLSTM)](https://arxiv.org/abs/2411.16422)

**Author:**
Abedin Sherifi

**Abstract:**
The aviation industry is rapidly evolving, driven by advancements in technology. Turbofan engines used in commercial aerospace are very complex systems. The majority of turbofan engine components are susceptible to degradation over the life of their operation. Turbofan engine degradation has an impact to engine performance, operability, and reliability. Predicting accurate remaining useful life (RUL) of a commercial turbofan engine based on a variety of complex sensor data is of paramount importance for the safety of the passengers, safety of flight, and for cost effective operations. That is why it is essential for turbofan engines to be monitored, controlled, and maintained. RUL predictions can either come from model-based or data-based approaches. The model-based approach can be very expensive due to the complexity of the mathematical models and the deep expertise that is required in the domain of physical systems. The data-based approach is more frequently used nowadays thanks to the high computational complexity of computers, the advancements in Machine Learning (ML) models, and advancements in sensors. This paper is going to be focused on Bi-Directional Long Short-Term Memory (BLSTM) models but will also provide a benchmark of several RUL prediction databased models. The proposed RUL prediction models are going to be evaluated based on engine failure prediction benchmark dataset Commercial Modular Aero-Propulsion System Simulation (CMAPSS). The CMAPSS dataset is from NASA which contains turbofan engine run to failure events.
       


### [Strengthening Power System Resilience to Extreme Weather Events Through Grid Enhancing Technologies](https://arxiv.org/abs/2411.16962)

**Author:**
Joseph Nyangon

**Abstract:**
Climate change significantly increases risks to power systems, exacerbating issues such as aging infrastructure, evolving regulations, cybersecurity threats, and fluctuating demand. This paper focuses on the utilization of Grid Enhancing Technologies (GETs) to strengthen power system resilience in the face of extreme weather events. GETs are pivotal in optimizing energy distribution, enabling predictive maintenance, ensuring reliable electricity supply, facilitating renewable energy integration, and automating responses to power instabilities and outages. Drawing insights from resilience theory, the paper reviews recent grid resilience literature, highlighting increasing vulnerabilities due to severe weather events. It demonstrates how GETs are crucial in optimizing smart grid operations, thereby not only mitigating climate-related impacts but also promoting industrial transformation.
  Keywords: Climate change, power systems, grid enhancing technologies (GETs), power system resilience, extreme weather
       


### [When IoT Meet LLMs: Applications and Challenges](https://arxiv.org/abs/2411.17722)

**Authors:**
Ibrahim Kok, Orhan Demirci, Suat Ozdemir

**Abstract:**
Recent advances in Large Language Models (LLMs) have positively and efficiently transformed workflows in many domains. One such domain with significant potential for LLM integration is the Internet of Things (IoT), where this integration brings new opportunities for improved decision making and system interaction. In this paper, we explore the various roles of LLMs in IoT, with a focus on their reasoning capabilities. We show how LLM-IoT integration can facilitate advanced decision making and contextual understanding in a variety of IoT scenarios. Furthermore, we explore the integration of LLMs with edge, fog, and cloud computing paradigms, and show how this synergy can optimize resource utilization, enhance real-time processing, and provide scalable solutions for complex IoT applications. To the best of our knowledge, this is the first comprehensive study covering IoT-LLM integration between edge, fog, and cloud systems. Additionally, we propose a novel system model for industrial IoT applications that leverages LLM-based collective intelligence to enable predictive maintenance and condition monitoring. Finally, we highlight key challenges and open issues that provide insights for future research in the field of LLM-IoT integration.
       


### [A Cloud-based Real-time Probabilistic Remaining Useful Life (RUL) Estimation using the Sequential Monte Carlo (SMC) Method](https://arxiv.org/abs/2411.17824)

**Authors:**
Karthik Reddy Lyathakula, Fuh-Gwo Yuan

**Abstract:**
The remaining useful life (RUL) estimation is an important metric that helps in condition-based maintenance. Damage data obtained from the diagnostics techniques are often noisy and the RUL estimated from the data is less reliable. Estimating the probabilistic RUL by quantifying the uncertainty in the predictive model parameters using the noisy data increases confidence in the predicted values. Uncertainty quantification methods generate statistical samples for the model parameters, that represent the uncertainty, by evaluating the predictive model several times. The computational time for solving a physics-based predictive model is significant, which makes the statistical techniques to be computationally expensive. It is essential to reduce the computational time to estimate the RUL in a feasible time. In this work, real-time probabilistic RUL estimation is demonstrated in adhesively bonded joints using the Sequential Monte Carlo (SMC) sampling method and cloud-based computations. The SMC sampling method is an alternative to traditional MCMC methods, which enables generating the statistical parameter samples in parallel. The parallel computational capabilities of the SMC methods are exploited by running the SMC simulation on multiple cloud calls. This approach is demonstrated by estimating fatigue RUL in the adhesively bonded joint. The accuracy of probabilistic RUL estimated by SMC is validated by comparing it with RUL estimated by the MCMC and the experimental values. The SMC simulation is run on the cloud and the computational speedup of the SMC is demonstrated.
       


## December
### [Prognostic Framework for Robotic Manipulators Operating Under Dynamic Task Severities](https://arxiv.org/abs/2412.00538)

**Authors:**
Ayush Mohanty, Jason Dekarske, Stephen K. Robinson, Sanjay Joshi, Nagi Gebraeel

**Abstract:**
Robotic manipulators are critical in many applications but are known to degrade over time. This degradation is influenced by the nature of the tasks performed by the robot. Tasks with higher severity, such as handling heavy payloads, can accelerate the degradation process. One way this degradation is reflected is in the position accuracy of the robot's end-effector. In this paper, we present a prognostic modeling framework that predicts a robotic manipulator's Remaining Useful Life (RUL) while accounting for the effects of task severity. Our framework represents the robot's position accuracy as a Brownian motion process with a random drift parameter that is influenced by task severity. The dynamic nature of task severity is modeled using a continuous-time Markov chain (CTMC). To evaluate RUL, we discuss two approaches -- (1) a novel closed-form expression for Remaining Lifetime Distribution (RLD), and (2) Monte Carlo simulations, commonly used in prognostics literature. Theoretical results establish the equivalence between these RUL computation approaches. We validate our framework through experiments using two distinct physics-based simulators for planar and spatial robot fleets. Our findings show that robots in both fleets experience shorter RUL when handling a higher proportion of high-severity tasks.
       


### [Sensor-Driven Predictive Vehicle Maintenance and Routing Problem with Time Windows](https://arxiv.org/abs/2412.04350)

**Authors:**
Iman Kazemian, Bahar Cavdar, Murat Yildirim

**Abstract:**
Advancements in sensor technology offer significant insights into vehicle conditions, unlocking new venues to enhance fleet operations. While current vehicle health management models provide accurate predictions of vehicle failures, they often fail to integrate these forecasts into operational decision-making, limiting their practical impact. This paper addresses this gap by incorporating sensor-driven failure predictions into a single-vehicle routing problem with time windows. A maintenance cost function is introduced to balance two critical trade-offs: premature maintenance, which leads to underutilization of remaining useful life, and delayed maintenance, which increases the likelihood of breakdowns. Routing problems with time windows are inherently challenging, and integrating maintenance considerations adds significantly to its computational complexity. To address this, we develop a new solution method, called the Iterative Alignment Method (IAM), building on the structural properties of the problem. IAM generates high-quality solutions even in large-size instances where Gurobi cannot find any solutions. Moreover, compared to the traditional periodic maintenance strategy, our sensor-driven approach to maintenance decisions shows improvements in operational and maintenance costs as well as in overall vehicle reliability.
       


### [Shifting NER into High Gear: The Auto-AdvER Approach](https://arxiv.org/abs/2412.05655)

**Authors:**
Filippos Ventirozos, Ioanna Nteka, Tania Nandy, Jozef Baca, Peter Appleby, Matthew Shardlow

**Abstract:**
This paper presents a case study on the development of Auto-AdvER, a specialised named entity recognition schema and dataset for text in the car advertisement genre. Developed with industry needs in mind, Auto-AdvER is designed to enhance text mining analytics in this domain and contributes a linguistically unique NER dataset. We present a schema consisting of three labels: "Condition", "Historic" and "Sales Options". We outline the guiding principles for annotation, describe the methodology for schema development, and show the results of an annotation study demonstrating inter-annotator agreement of 92% F1-Score. Furthermore, we compare the performance by using encoder-only models: BERT, DeBERTaV3 and decoder-only open and closed source Large Language Models (LLMs): Llama, Qwen, GPT-4 and Gemini. Our results show that the class of LLMs outperforms the smaller encoder-only models. However, the LLMs are costly and far from perfect for this task. We present this work as a stepping stone toward more fine-grained analysis and discuss Auto-AdvER's potential impact on advertisement analytics and customer insights, including applications such as the analysis of market dynamics and data-driven predictive maintenance. Our schema, as well as our associated findings, are suitable for both private and public entities considering named entity recognition in the automotive domain, or other specialist domains.
       


### [Digital Transformation in the Water Distribution System based on the Digital Twins Concept](https://arxiv.org/abs/2412.06694)

**Authors:**
MohammadHossein Homaei, Agustín Javier Di Bartolo, Mar Ávila, Óscar Mogollón-Gutiérrez, Andrés Caro

**Abstract:**
Digital Twins have emerged as a disruptive technology with great potential; they can enhance WDS by offering real-time monitoring, predictive maintenance, and optimization capabilities. This paper describes the development of a state-of-the-art DT platform for WDS, introducing advanced technologies such as the Internet of Things, Artificial Intelligence, and Machine Learning models. This paper provides insight into the architecture of the proposed platform-CAUCCES-that, informed by both historical and meteorological data, effectively deploys AI/ML models like LSTM networks, Prophet, LightGBM, and XGBoost in trying to predict water consumption patterns. Furthermore, we delve into how optimization in the maintenance of WDS can be achieved by formulating a Constraint Programming problem for scheduling, hence minimizing the operational cost efficiently with reduced environmental impacts. It also focuses on cybersecurity and protection to ensure the integrity and reliability of the DT platform. In this view, the system will contribute to improvements in decision-making capabilities, operational efficiency, and system reliability, with reassurance being drawn from the important role it can play toward sustainable management of water resources.
       


### [Intelligent Electric Power Steering: Artificial Intelligence Integration Enhances Vehicle Safety and Performance](https://arxiv.org/abs/2412.08133)

**Authors:**
Vikas Vyas, Sneha Sudhir Shetiya

**Abstract:**
Electric Power Steering (EPS) systems utilize electric motors to aid users in steering their vehicles, which provide additional precise control and reduced energy consumption compared to traditional hydraulic systems. EPS technology provides safety,control and efficiency.. This paper explains the integration of Artificial Intelligence (AI) into Electric Power Steering (EPS) systems, focusing on its role in enhancing the safety, and adaptability across diverse driving conditions. We explore significant development in AI-driven EPS, including predictive control algorithms, adaptive torque management systems, and data-driven diagnostics. The paper presents case studies of AI applications in EPS, such as Lane centering control (LCC), Automated Parking Systems, and Autonomous Vehicle Steering, while considering the challenges, limitations, and future prospects of this technology. This article discusses current developments in AI-driven EPS, emphasizing on the benefits of improved safety, adaptive control, and predictive maintenance. Challenges in integrating AI in EPS systems. This paper addresses cybersecurity risks, ethical concerns, and technical limitations,, along with next steps for research and implementation in autonomous, and connected vehicles.
       


### [Elastic Modulus Versus Cell Packing Density in MDCK Epithelial Monolayers](https://arxiv.org/abs/2412.09443)

**Authors:**
Steven J. Chisolm, Emily Guo, Vignesh Subramaniam, Kyle D. Schulze, Thomas E. Angelini

**Abstract:**
The elastic moduli of tissues are connected to their states of health and function. The epithelial monolayer is a simple, minimal, tissue model that is often used to gain understanding of mechanical behavior at the cellular or multi-cellular scale. Here we investigate how the elastic modulus of Madin Darby Canine Kidney (MDCK) cells depends on their packing density. Rather than measuring elasticity at the sub-cellular scale with local probes, we characterize the monolayer at the multi-cellular scale, as one would a thin slab of elastic material. We use a micro-indentation system to apply gentle forces to the apical side of MDCK monolayers, applying a normal force to approximately 100 cells in each experiment. In low-density confluent monolayers, we find that the elastic modulus decreases with increasing cell density. At high densities, the modulus appears to plateau. This finding will help guide our understanding of known collective behaviors in epithelial monolayers and other tissues where variations in cell packing density are correlated with cell motion.
       


### [A novel ML-fuzzy control system for optimizing PHEV fuel efficiency and extending electric range under diverse driving conditions](https://arxiv.org/abs/2412.09499)

**Authors:**
Mehrdad Raeesi, Saba Mansour, Sina Changizian

**Abstract:**
Aiming for a greener transportation future, this study introduces an innovative control system for plug-in hybrid electric vehicles (PHEVs) that utilizes machine learning (ML) techniques to forecast energy usage in the pure electric mode of the vehicle and optimize power allocation across different operational modes, including pure electric, series hybrid, parallel hybrid, and internal combustion operation. The fuzzy logic decision-making process governs the vehicle control system. The performance was assessed under various driving conditions. Key findings include a significant enhancement in pure electric mode efficiency, achieving an extended full-electric range of approximately 84 kilometers on an 80% utilization of a 20-kWh battery pack. During the WLTC driving cycle, the control system reduced fuel consumption to 2.86 L/100km, representing a 20% reduction in gasoline-equivalent fuel consumption. Evaluations of vehicle performance at discrete driving speeds, highlighted effective energy management, with the vehicle battery charging at lower speeds and discharging at higher speeds, showing optimized energy recovery and consumption strategies. Initial battery charge levels notably influenced vehicle performance. A 90% initial charge enabled prolonged all-electric operation, minimizing fuel consumption to 2 L/100km less than that of the base control system. Real-world driving pattern analysis revealed significant variations, with shorter, slower cycles requiring lower fuel consumption due to prioritized electric propulsion, while longer, faster cycles increased internal combustion engine usage. The control system also adapted to different battery state of health (SOH) conditions, with higher SOH facilitating extended electric mode usage, reducing total fuel consumption by up to 2.87 L/100km.
       


### [Data-Driven Quantification of Battery Degradation Modes via Critical Features from Charging](https://arxiv.org/abs/2412.10044)

**Authors:**
Yuanhao Cheng, Hanyu Bai, Yichen Liang, Xiaofan Cui, Weiren Jiang, Ziyou Song

**Abstract:**
Battery degradation modes influence the aging behavior of Li-ion batteries, leading to accelerated capacity loss and potential safety issues. Quantifying these aging mechanisms poses challenges for both online and offline diagnostics in charging station applications. Data-driven algorithms have emerged as effective tools for addressing state-of-health issues by learning hard-to-model electrochemical properties from data. This paper presents a data-driven method for quantifying battery degradation modes. Ninety-one statistical features are extracted from the incremental capacity curve derived from 1/3C charging data. These features are then screened based on dispersion, contribution, and correlation. Subsequently, machine learning models, including four baseline algorithms and a feedforward neural network, are used to estimate the degradation modes. Experimental validation indicates that the feedforward neural network outperforms the others, achieving a root mean square error of around 10\% across all three degradation modes (i.e., loss of lithium inventory, loss of active material on the positive electrode, and loss of active material on the negative electrode). The findings in this paper demonstrate the potential of machine learning for diagnosing battery degradation modes in charging station scenarios.
       


### [Transformer-Based Bearing Fault Detection using Temporal Decomposition Attention Mechanism](https://arxiv.org/abs/2412.11245)

**Authors:**
Marzieh Mirzaeibonehkhater, Mohammad Ali Labbaf-Khaniki, Mohammad Manthouri

**Abstract:**
Bearing fault detection is a critical task in predictive maintenance, where accurate and timely fault identification can prevent costly downtime and equipment damage. Traditional attention mechanisms in Transformer neural networks often struggle to capture the complex temporal patterns in bearing vibration data, leading to suboptimal performance. To address this limitation, we propose a novel attention mechanism, Temporal Decomposition Attention (TDA), which combines temporal bias encoding with seasonal-trend decomposition to capture both long-term dependencies and periodic fluctuations in time series data. Additionally, we incorporate the Hull Exponential Moving Average (HEMA) for feature extraction, enabling the model to effectively capture meaningful characteristics from the data while reducing noise. Our approach integrates TDA into the Transformer architecture, allowing the model to focus separately on the trend and seasonal components of the data. Experimental results on the Case Western Reserve University (CWRU) bearing fault detection dataset demonstrate that our approach outperforms traditional attention mechanisms and achieves state-of-the-art performance in terms of accuracy and interpretability. The HEMA-Transformer-TDA model achieves an accuracy of 98.1%, with exceptional precision, recall, and F1-scores, demonstrating its effectiveness in bearing fault detection and its potential for application in other time series tasks with seasonal patterns or trends.
       


### [Multimodal LLM for Intelligent Transportation Systems](https://arxiv.org/abs/2412.11683)

**Authors:**
Dexter Le, Aybars Yunusoglu, Karn Tiwari, Murat Isik, I. Can Dikmen

**Abstract:**
In the evolving landscape of transportation systems, integrating Large Language Models (LLMs) offers a promising frontier for advancing intelligent decision-making across various applications. This paper introduces a novel 3-dimensional framework that encapsulates the intersection of applications, machine learning methodologies, and hardware devices, particularly emphasizing the role of LLMs. Instead of using multiple machine learning algorithms, our framework uses a single, data-centric LLM architecture that can analyze time series, images, and videos. We explore how LLMs can enhance data interpretation and decision-making in transportation. We apply this LLM framework to different sensor datasets, including time-series data and visual data from sources like Oxford Radar RobotCar, D-Behavior (D-Set), nuScenes by Motional, and Comma2k19. The goal is to streamline data processing workflows, reduce the complexity of deploying multiple models, and make intelligent transportation systems more efficient and accurate. The study was conducted using state-of-the-art hardware, leveraging the computational power of AMD RTX 3060 GPUs and Intel i9-12900 processors. The experimental results demonstrate that our framework achieves an average accuracy of 91.33\% across these datasets, with the highest accuracy observed in time-series data (92.7\%), showcasing the model's proficiency in handling sequential information essential for tasks such as motion planning and predictive maintenance. Through our exploration, we demonstrate the versatility and efficacy of LLMs in handling multimodal data within the transportation sector, ultimately providing insights into their application in real-world scenarios. Our findings align with the broader conference themes, highlighting the transformative potential of LLMs in advancing transportation technologies.
       


### [Harnessing Event Sensory Data for Error Pattern Prediction in Vehicles: A Language Model Approach](https://arxiv.org/abs/2412.13041)

**Authors:**
Hugo Math, Rainer Lienhart, Robin Schön

**Abstract:**
In this paper, we draw an analogy between processing natural languages and processing multivariate event streams from vehicles in order to predict $\textit{when}$ and $\textit{what}$ error pattern is most likely to occur in the future for a given car. Our approach leverages the temporal dynamics and contextual relationships of our event data from a fleet of cars. Event data is composed of discrete values of error codes as well as continuous values such as time and mileage. Modelled by two causal Transformers, we can anticipate vehicle failures and malfunctions before they happen. Thus, we introduce $\textit{CarFormer}$, a Transformer model trained via a new self-supervised learning strategy, and $\textit{EPredictor}$, an autoregressive Transformer decoder model capable of predicting $\textit{when}$ and $\textit{what}$ error pattern will most likely occur after some error code apparition. Despite the challenges of high cardinality of event types, their unbalanced frequency of appearance and limited labelled data, our experimental results demonstrate the excellent predictive ability of our novel model. Specifically, with sequences of $160$ error codes on average, our model is able with only half of the error codes to achieve $80\%$ F1 score for predicting $\textit{what}$ error pattern will occur and achieves an average absolute error of $58.4 \pm 13.2$h $\textit{when}$ forecasting the time of occurrence, thus enabling confident predictive maintenance and enhancing vehicle safety.
       


### [Switching Frequency as FPGA Monitor: Studying Degradation and Ageing Prognosis at Large Scale](https://arxiv.org/abs/2412.15720)

**Authors:**
Leandro Lanzieri, Lukasz Butkowski, Jiri Kral, Goerschwin Fey, Holger Schlarb, Thomas C. Schmidt

**Abstract:**
The growing deployment of unhardened embedded devices in critical systems demands the monitoring of hardware ageing as part of predictive maintenance. In this paper, we study degradation on a large deployment of 298 naturally aged FPGAs operating in the European XFEL particle accelerator. We base our statistical analyses on 280 days of in-field measurements and find a generalized and continuous degradation of the switching frequency across all devices with a median value of 0.064%. The large scale of this study allows us to localize areas of the deployed FPGAs that are highly impacted by degradation. Moreover, by training machine learning models on the collected data, we are able to forecast future trends of frequency degradation with horizons of 60 days and relative errors as little as 0.002% over an evaluation period of 100 days.
       


### [CNN-LSTM Hybrid Deep Learning Model for Remaining Useful Life Estimation](https://arxiv.org/abs/2412.15998)

**Authors:**
Muthukumar G, Jyosna Philip

**Abstract:**
Remaining Useful Life (RUL) of a component or a system is defined as the length from the current time to the end of the useful life. Accurate RUL estimation plays a crucial role in Predictive Maintenance applications. Traditional regression methods, both linear and non-linear, have struggled to achieve high accuracy in this domain. While Convolutional Neural Networks (CNNs) have shown improved accuracy, they often overlook the sequential nature of the data, relying instead on features derived from sliding windows. Since RUL prediction inherently involves multivariate time series analysis, robust sequence learning is essential. In this work, we propose a hybrid approach combining Convolutional Neural Networks with Long Short-Term Memory (LSTM) networks for RUL estimation. Although CNN-based LSTM models have been applied to sequence prediction tasks in financial forecasting, this is the first attempt to adopt this approach for RUL estimation in prognostics. In this approach, CNN is first employed to efficiently extract features from the data, followed by LSTM, which uses these extracted features to predict RUL. This method effectively leverages sensor sequence information, uncovering hidden patterns within the data, even under multiple operating conditions and fault scenarios. Our results demonstrate that the hybrid CNN-LSTM model achieves the highest accuracy, offering a superior score compared to the other methods.
       


### [RUL forecasting for wind turbine predictive maintenance based on deep learning](https://arxiv.org/abs/2412.17823)

**Authors:**
Syed Shazaib Shah, Tan Daoliang, Sah Chandan Kumar

**Abstract:**
Predictive maintenance (PdM) is increasingly pursued to reduce wind farm operation and maintenance costs by accurately predicting the remaining useful life (RUL) and strategically scheduling maintenance. However, the remoteness of wind farms often renders current methodologies ineffective, as they fail to provide a sufficiently reliable advance time window for maintenance planning, limiting PdM's practicality. This study introduces a novel deep learning (DL) methodology for future RUL forecasting. By employing a multi-parametric attention-based DL approach that bypasses feature engineering, thereby minimizing the risk of human error, two models: ForeNet-2d and ForeNet-3d are proposed. These models successfully forecast the RUL for seven multifaceted wind turbine (WT) failures with a 2-week forecast window. The most precise forecast deviated by only 10 minutes from the actual RUL, while the least accurate prediction deviated by 1.8 days, with most predictions being off by only a few hours. This methodology offers a substantial time frame to access remote WTs and perform necessary maintenance, thereby enabling the practical implementation of PdM.
       


### [Data-driven tool wear prediction in milling, based on a process-integrated single-sensor approach](https://arxiv.org/abs/2412.19950)

**Authors:**
Eric Hirsch, Christian Friedrich

**Abstract:**
Accurate tool wear prediction is essential for maintaining productivity and minimizing costs in machining. However, the complex nature of the tool wear process poses significant challenges to achieving reliable predictions. This study explores data-driven methods, in particular deep learning, for tool wear prediction. Traditional data-driven approaches often focus on a single process, relying on multi-sensor setups and extensive data generation, which limits generalization to new settings. Moreover, multi-sensor integration is often impractical in industrial environments. To address these limitations, this research investigates the transferability of predictive models using minimal training data, validated across two processes. Furthermore, it uses a simple setup with a single acceleration sensor to establish a low-cost data generation approach that facilitates the generalization of models to other processes via transfer learning. The study evaluates several machine learning models, including convolutional neural networks (CNN), long short-term memory networks (LSTM), support vector machines (SVM) and decision trees, trained on different input formats such as feature vectors and short-time Fourier transform (STFT). The performance of the models is evaluated on different amounts of training data, including scenarios with significantly reduced datasets, providing insight into their effectiveness under constrained data conditions. The results demonstrate the potential of specific models and configurations for effective tool wear prediction, contributing to the development of more adaptable and efficient predictive maintenance strategies in machining. Notably, the ConvNeXt model has an exceptional performance, achieving an 99.1% accuracy in identifying tool wear using data from only four milling tools operated until they are worn.
       


