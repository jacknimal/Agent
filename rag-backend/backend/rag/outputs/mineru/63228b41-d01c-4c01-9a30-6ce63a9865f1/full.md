# PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization

Xiaoya Cheng1 Long Wang2,3 Yan Liu4Xinyi Liu1 Hanlin Tan1 Yu Liu1Maojun Zhang1 Shen Yan1t

1National University of Defense Technology²Zhejiang University

3Westlake University4Hangzhou Dianzi University

{chengxy，liuxinyi24， hanlin\_tan， jasonyuliu， mjzhang， yanshenl2}@nudt.edu.cnwanglong@westlake.edu.cn43038@hdu.edu.cn

https://nudt-sawlab.github.io/PiLoT/

![](images/5e6a9a7e89e2c9067127859cb396910d0b67fc1c5ece1394636be190c7556f67.jpg)  
Figure 1. Overview of PiLoT.Our system takes a live video frame and a geo-referenced 3D map as input,and outputs 1) the UAV 6-DoF pose,visualized by the tight alignment in the AR overlays (bottom row),and 2) the 3D geo-coordinates of any target pixel,as shown in the dynamic target tracking example (left, filmstrip view). PiLoT achieves drift-free, real-time,and long-term ego and target geo-localization, demonstrated on a 10 km UAV trajectory with error color-coded (green: low,red: high). The system attains a median error of 1.37 m, a per-frame latency of 30 \~ 40 ms,and 1Oo% success rate across day-to-night and cross-season variations without GNSS and IMU signals.

## Abstract

Wepresent PiLoT,a unified framework that tackles UAVbased ego and target geo-localization. Conventional approaches rely on decoupled pipelines that fuse GNsS and Visual-Inertial Odometry(ViO) for ego-pose estimation, and active sensors like laser rangefinders for target localization. However, these methods are susceptible to failure in GNSS-denied environments and incur substantial hardware costs and complexity. PiLoT breaks this paradigm by directly registering live video stream against a georeferenced 3D map. To achieve robust, accurate, and realtime performance,we introduce three key contributions:

1） a Dual-Thread Engine that decouples map rendering from core localization thread, ensuring both low latency while maintaining drift-free accuracy; 2) a large-scale synthetic dataset with precise geometric annotations (camera pose,depth maps). This dataset enables the training of a lightweight network that generalizes in a zero-shot manner from simulation to real data; and 3) a Joint Neural-Guided Stochastic-Gradient Optimizer(JNGO) that achieves robust convergence even under aggressive motion.Evaluations on comprehensive public and newly collected benchmarks show that PiLoToutperforms state-of-the-art methods while running over 25 FPS on NVIDIA Jetson Orin platform.

## 1. Introduction

If a UAV with a single monocular camera could instantly localize itself in the world and geolocate everything it sees, it would unlock a new era of autonomy, enabling true-to-life digital twins,AR/VR applications,reliable navigation,and embodied AI for UAV. The mainstream approaches 6, 24] tackle the problem with a decoupled pipeline: localizing the UAV using visual-inertial odometry(VIO） fused with GNSS,and subsequently employing active sensors like laser rangefinders for target acquisition (e.g., DJI Matrice 4 Series).However, this paradigm suffers from two critical limitations:its reliance on GNSS makes it fragile in denied or degraded environments,and its laser-based targeting is costly, cumbersome,and restricted to a single point.

In this paper, we argue for a fundamental paradigm shift away from these decoupled, sensor-heavy systems. Our core idea is to reformulate UAV-based ego and target geo-localization as a unified pixel-to-3D registration problem.By continuously registering the live UAV video stream against a global 3D map (Google Earth for example), our system inherently recovers the UAV 6-DoF pose and geocoordinates of any given pixel in the query image.

However, realizing such a system is non-trivial, as it requires resolving the fundamental trade-offs between accuracy,robustness and real-time performance,often termed the “impossible triangle". This challenge is profoundly amplified by the demanding conditions of aerial deployment, specifically: 1) Drift-Free Accuracy. While VIO/SLAM methods [5,6,21,24] could provide smooth localization, they inherently accumulate drift over long-duration flights. 2）Environmental and Motion Robustness. Localization is challenged by severe appearance variations (e.g., day-to-night, seasonal changes) between the captured video and the reference map. Concurrently,aggressive 6-DoF UAV motion can cause large inter-frame displacements, exceeding the basin of convergence of standard optimizers. 3） Real-time Performance. Recovering a globally consistent pose per frame is computationally intensive,creating a significant bottleneck for learning-based matchers [9, 23, 25, 27, 29, 38] on resource-constrained onboard hardware (e.g., NVIDIA Jetson Orin).

To conquer these challenges,we propose PiLoT,a new paradigm for UAV-based ego and target localization. Pi-LoT is built upon three key technical contributions, each designed to address a side of the impossible triangle. First, we introduce a dual-thread framework to decouple localization from map rendering. A Render Thread generates a geo-referenced synthetic view on-the-fly，while a concurrent Localization Thread registers the incoming video stream against this view in a feature space. This design ensures that every query frame is constrained by dynamically updated geo-anchors that follows the UAV's perspective, which is the key to achieve drift-free performance.

Second, we develop a custom AirSim-Cesium-Unreal engine simulator,and build a new, million-scale synthetic dataset by simulating UAV trajectories over vast photorealistic global terrains.By providing calibrated geometric supervision (metric depth,verified poses) across diverse conditions (weather, lighting),our dataset compels the network to learn features grounded in stable 3D geometry. Experiments confirm that these domain-invariant geometric cues enable our UAV-specific network to achieve zero-shot generalization on real-world data.

Third,we propose a Joint Neural-Guided Stochastic-Gradient Optimizer (JNGO) that performs pixel-to-3D registration of the query frame. At its core, JNGO optimizes the camera pose by aligning the query frame with the pro-jection of a reference 3D map in a shared feature space. To handle aggressive UAV motions, JNGO synergizes stochastic and gradient-based optimization for effective global exploration and local refinement. Specifically,it first generates a multitude of initial pose hypotheses. Each hypothesis is then refined in parallel via a gradient-based optimization that maximizes feature alignment, an operation we efficiently implement in CUDA.By repeating this process across multiple feature levels, JNGO achieves robust, realtime convergence even under extreme inter-frame displacements of up to 10 meters and 10 degrees of yaw.

To validate our approach and beneft the research community,we introduce a new, comprehensive benchmark suite for UAV-based ego and target geo-localization. Our benchmark comprises both challenging real-world sequences and large-scale synthetic data, specifically designed to test robustness against severe weather and lighting variations,aggressive motion,and long-term flights. Extensive experiments on public dataset and this benchmark demonstrate that our method substantially outperforms state-of-the-art vision-based approaches in both ego and target geo-localization. Furthermore,we validate its practical viability by deploying it on an NVIDIA Jetson Orin, where it achieves real-time performance with 25 FPS.

## 2. Related Work

UAV-based Ego localization. UAV-based ego localization aims to estimate the 6-DoF pose in a global coordinate. While SLAM and VIO methods [5,21] are robust for local state estimation,they are prone to drift in the absence of a global reference,limiting their universal applicability for UAVs. To achieve drift-free localization,a dominant paradigm is to register the UAV's view against a georeferenced map.

Early methods used 2D satellite imagery [19,28, 32], yielding only 3-DoF pose (latitude, longitude,and yaw), and remain limited to 2D maps and simplified top-down assumptions．To recover the full 6-DoF pose,recent works have turned to 3D maps [23,25,27,38]. These methods typically initialize the pose via retrieval [1,3,12] or sensor priors,and then refine it through either matchingbased [9,20,26,29,41] or direct alignment approaches [13,16,22, 27,39]. Matching-based methods are computationally too expensive and time-consuming for resourceconstrained UAVs [9,20,26,29,41]. Alternatively， direct alignment methods,such as photometric optimization [16,22,39],are highly sensitive to outdoor illumination,whereas feature-metric optimization provides better robustness but remains initialization-sensitive and generalizes poorly to aerial UAV views [13,27]. Our unified pixel-to-3D framework mitigates initialization sensitivity and closes the aerial domain gap via large-scale, mapgrounded training.

Table1.Comparison of UAV-oriented datasets．Legend: $\checkmark =$ yes, ${ \cal X } = { \bf n } { \bf o } .$ Cond.= illumination/weather changes; $6 { - } D o F =$ availability of ground-truth 6-DoF pose;Depth = availability of metric depth; Seq. = contains continuous video sequences.
<table><tr><td>Dataset</td><td>Img</td><td>Region</td><td>Alt (m)</td><td>Pitch (°)</td><td>Type</td><td></td><td>Cond. 6-DoF Depth Seq.</td><td></td><td></td></tr><tr><td>University-1652 [40]</td><td>37.8k</td><td>1,652</td><td>-</td><td>varied</td><td>Synthetic</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>SUES-200 [43]</td><td>24.1k</td><td>200</td><td>150-300</td><td>varied</td><td>Real</td><td>X</td><td>X</td><td>×</td><td>√</td></tr><tr><td>DenseUAV [8]</td><td>27k</td><td>14</td><td>80-100</td><td>fixed</td><td>Real</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>UAV-VisLoc [37]</td><td>6.7k</td><td>11</td><td>400-2000</td><td>fixed</td><td>Real</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>UAVD4L [36]</td><td>6.8k</td><td>1</td><td>50-300</td><td>varied</td><td>Real</td><td>X</td><td>√</td><td>√</td><td>X</td></tr><tr><td>GAME4Loc [15]</td><td>33.7k</td><td>1</td><td>80-650</td><td>varied</td><td>Synthetic</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>MatrixCity [17]</td><td>67k</td><td>2</td><td>100-450</td><td>fixed</td><td>Synthetic</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>OrthoLoC[10]</td><td>16.4k</td><td>47</td><td>50-300</td><td>varied</td><td>Real</td><td>X</td><td>√</td><td>√</td><td>X</td></tr><tr><td>UAVScenes [34]</td><td>120k</td><td>4</td><td>80-130</td><td>fixed</td><td>Real</td><td>X</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Ours</td><td>1M+</td><td>378</td><td>~800</td><td>varied</td><td>Synthetic</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

UAV-based Target Geo-localization. UAV-based target geo-localization determines the 3D world coordinates of a target from an image. A primary approach relies on geometric principles,combining camera models with UAV poses to infer target coordinates [4,42]． This method is highly sensitive to the quality of the pose estimate. To im-prove accuracy, 3D map-based geolocation [36] has become a mainstream approach,using techniques like renderingbased matching and DSM projection to obtain precise target coordinates. While accurate,these methods are often bottlenecked by the need for computationally expensive rendering-based matching or tight coupling with a non-realtime ego-localization pipeline. This prevents them from achieving the millisecond-level response required for dynamic targeting applications, a gap our work aims to fill.

UAV-Oriented Visual Localization Datasets. Although recent visual localization methods have made significant progress, they remain largely tailored to ground-level tasks. Models trained on such data often fail in aerial domains, highlighting the need for UAV specific datasets. As detailed in Tab. 1, while many [8,15,17,34,36,37,40,43] offer photorealism, they often lack the scale and complete geometric ground truth (e.g., 6-DoF poses,metric depth) needed for robust training. Even recent efforts with geometric data [17,34] are limited in city scale and viewpoint variety,making them insufficient for sequential geometric supervision.To address the lack of structured UAV training data,we build a fully automated AirSim-Cesium-Unreal pipeline that converts large-scale geospatial data into geoaligned imagery with precise 6-DoF poses and depth maps.

![](images/3fa11e0fa130afc3d2b0ddd6615da10a4c762c53fa73ab8cd22ea571f3745e32.jpg)  
Figure 2.PiLoT's Dual-Thread Framework. We decouple rendering from localization into two parallel threads.A Render Thread generates synthetic views,while a concurrent Localization Thread registers the live frame against them to compute the pose, ensuring high-frequency accuracy.

## 3. Method

## 3.1. Overview

Given a geo-referenced 3D map $\mathcal { M } ,$ a monocular video stream $\{ I _ { i } ^ { q } \}$ with known intrinsics $\{ { \bf K } _ { i } \}$ ,and a single pose prior for the first frame $( \tilde { \mathbf { T } } _ { i n i t } )$ ，we address the problem of UAV-based ego localization and target geo-localization without aid from GNSS and IMU. Specifically, our goals are twofold: 1) estimate T; for every query frame,and 2) enable precise pixel-to-geo projection that maps any query pixel $\mathbf { u } = ( u , v ) ^ { \top }$ on frame $\{ I _ { i } ^ { q } \}$ to its real-world coordinates (lon,lat,alt).

## 3.2.PiLoT's Dual-Thread Framework

For sequential video localization, a naive strategy is to render a reference view from the last frame and perform pose refinement for the current frame. This linear dependency creates an inherent temporal bottleneck, where the localization engine is forced to stall until the rendering task completes.Instead of this conventional linear pipeline,we propose a decoupled dual-thread architecture that synchronizes map rendering and pose optimization in parallel (Fig. 2).

Within this framework,a common strategy to handle rapid motion is to render multiple views around the last estimated pose, thereby expanding the search region. Our system instead strategically renders a single reference anchor and leverages a one-to-many strategy. This approach refines a swarm of pose hypotheses against the shared rendering,achieving a wide search range without multiple reference viewpoints. The two threads coordinate this process as follows:

Rendering thread. This thread runs to provide a georeferenced view for localization. As illustrated in Fig.2 (bottom),the rendering thread first predicts a reference pose $\hat { \mathbf { T } } _ { i \left. i - 1 \right. }$ from the last estimate $\hat { \mathbf { T } } _ { i - 1 }$ using a constantvelocity Kalman filter (KF). From this predicted pose, we then render a new reference view $( I _ { i } ^ { r } , D _ { i } ^ { r } )$ and back-project $N$ of its depth-valid pixels into the world frame to form a set of 3D geo-anchors:

![](images/c75dc5c49e97bb7462f0d3da654229b88f89580b49408dcb9ce9505e58887e8a.jpg)  
Figure 3. Overview of the PiLoT framework and localization pipeline. (a) The overal pipeline inputs a query frame and outputs the UAV's 6-DoF ego-pose along with the target's 3-DoF geo-location. (b) A highly efficient one-to-many paradigm matches multiple query hypotheses against a single rendered reference view via feature alignment.(c) Our coarse-to-fine optimizer iteratively narrows the search space to converge on the optimal 6-DoF pose. (d) The final estimated trajectory demonstrates robust and drift-free sequential localization.

$$
\mathbf P _ { i , j } ^ { W } = \hat { \mathbf T } _ { i | i - 1 } \left( D _ { i } ^ { r } ( \mathbf p _ { i , j } ^ { r } ) \cdot \mathbf K ^ { - 1 } \mathbf p _ { i , j } ^ { r } \right)\tag{1}
$$

We finally package a reference bundle

$$
\boldsymbol { B _ { i } } \ : = \ \left( I _ { i } ^ { r } , \hat { \mathbf { T } } _ { i | i - 1 } , \{ \mathbf { P } _ { i , j } ^ { W } \} _ { j = 1 } ^ { N } \right)\tag{2}
$$

and pass it to the localization thread.

Localization thread.As depicted in Fig.2 (top), our Pixelto-3D Registration pipeline executes for each a new query frame $I _ { i + 1 } ^ { q }$ It begins by extracting multi-scale features and uncertainty maps from both the query and the reference view $I _ { i } ^ { r }$ using a lightweight extractor (Sec. 3.3.1). Anchored by the reference bundle $B _ { i } .$ ，our JNGO optimizer (Sec.3.3.2) then performs a global exploration with local exploitation to conduct a wide-area search and find the globally consistent pose estimate $\hat { \mathbf { T } } _ { i + 1 }$ . This new pose is subsequently passed back to the rendering thread to prepare the reference bundle $B _ { i + : }$ for the next cycle.

For the sake of readability,we will omit the frame index i from our notations in the following sections when the context is clear, for example,using $P _ { j } ^ { W }$ instead of $P _ { i , j } ^ { W }$

## 3.3. Pixel-to-3D Registration

## 3.3.1UAV-specific Feature Extraction

Lightweight neural network.We seek UAV-specific features that remain discriminative under large viewpoint and ilumination changes while running at edge speed.We adopt an off-the-shelf MobileOne-SO encoder (depth=3, ImageNet-initialized) with a compact U-Net decoder, shared by the query Iq and reference Ir branches. Given an $H \times W$ RGB image, it outputs a three-level pyramid at 1/4 (coarse), $1 / 2$ (mid),and 1 (fine) resolution with a compact channel width $C { = } 3 2 .$ ,yielding the query features and uncertainties $\{ ( \mathbf { f } _ { \ell } ^ { q } , \boldsymbol { w } _ { \ell } ^ { q } ) \} _ { \ell = 0 } ^ { 2 }$ and the reference counterparts $\{ ( \mathbf { f } _ { \ell } ^ { r } , w _ { \ell } ^ { r } ) \} _ { \ell = 0 } ^ { 2 } .$ For method details,please refer to Sec.A.1 in the Appendix.

Training with a large-scale dataset. Training this neural network hinges on large-scale datasets with dense depth and precise camera poses for geometric supervision.However, existing UAV datasets [8,15,40, 43] are deficient in such labels and scale. Even recent efforts with geometric data [17, 34] are limited in a few city models and viewpoint variety,making them insufficient for sequential geometric supervision. To bridge this critical data gap,we introduce a new large-scale synthetic dataset specifically designed to support geometry-aware learning.

We develop a fully automated simulator based on the AirSim-Cesium-Unreal Engine pipeline. Using this powerful tool,we generate a new,million-scale synthetic dataset by simulating flights over vast, photorealistic global terrains.As illustrated in Fig. 4(a-c), our dataset provides RGB and pixel-wise depth images captured along realistic UAV trajectories under diverse visual conditions (e.g., scenes,weather, lighting). Crucially,we provide precise and geometrically-consistent ground truth, including absolute camera poses,all rigorously validated through repro-jection.Please see Sec.B.1 in the Appendix for details on our dataset generation and statistics.

By providing accurate geometric supervision across diverse visual conditions,our large-scale dataset compels the network to learn features grounded in the underlying 3D structure. This makes the learned representations inherently robust to photometric variations and is the key to training a lightweight UAV-specific network that achieves zero-shot generalization to real-world data,as illustrated in Fig.4(d). Supervision. We train our network using a direct alignment approach, jointly optimizing the feature extractor and the subsequent iterative pose refinement process end-toend. The core of our training is a geometric loss that minimizes the reprojection error between the ground-truth 2D projections $\mathbf { p } _ { j } ^ { q }$ of the geo-anchors and their estimated pro-jections $\tilde { \mathbf { p } } _ { j } ^ { q }$ based on the pose estimate:

![](images/7a0d4efcb565710565c628af66f3f3bf12cad04f1d4ecaa8dc5ee5aa35448f78.jpg)  
Figure 4. Overview of our synthetic data generation and its resulting zero-shot sim-to-real performance. From left to right: (a) realistic UAV trajectories rendered over geo-referenced 3D tiles in Cesium for Unreal; (b) multi-condition diversity across weather/time and viewpoint (in-plane yaw, out-of-plane pitch/yaw, planar translation $T _ { x } , T _ { y } ,$ altitude $T _ { z } ) ;$ (c) geometric consistency:we export absolute per-pixel depth and validate by reprojection; (d) our three-level feature pyramid on query (real) vs. reference (synthetic) images.

$$
\mathcal { L } = \sum _ { j } \rho _ { B } \left( \left| \left| \mathbf { p } _ { j } ^ { q } - \tilde { \mathbf { p } } _ { j } ^ { q } \right| \right| _ { 2 } ^ { 2 } \right)\tag{3}
$$

where $\rho _ { B } ( \cdot )$ is Barron's robust loss function [2].

## 3.3.2Joint Neural-Guided Stochastic-Gradient Optimizer

Aggressive UAV motion often induces large inter-frame displacements,posing a significant challenge for traditional gradient-based optimizers that are prone to local minima. To address this,we introduce the JNGO,which navigates the challenging, non-convex optimization landscape by synergizing a global exploration with local exploitation.

Rotation-Aware Hypothesis Generation. Based on the observation that apparent pixel displacement in UAV imagery is far more sensitive to rotations than to translations, we design a Rotation-Aware Sampling strategy to generate the hypotheses $\tilde { \mathbf { T } } _ { m }$ by adaptively enlarges the search range along motion-sensitive axes, pitch and yaw. As shown in Fig. 5 (a),centered at the hypotheses on the previous frame,rotational perturbations are sampled uniformly from an anisotropic bounding box $B _ { r }$ that allocates greater range to pitch and yaw:

$$
\mathcal { B } _ { r } = [ - \alpha _ { \mathrm { p i t c h } } , \alpha _ { \mathrm { p i t c h } } ] \times [ - \alpha _ { \mathrm { y a w } } , \alpha _ { \mathrm { y a w } } ]\tag{4}
$$

and minor translational perturbations are drawn from a

Gaussian distribution:

$$
\begin{array} { l } { \displaystyle \delta \mathbf { t } _ { m } \sim \mathcal { N } \big ( \pmb { \mu } _ { t } , \pmb { \Sigma } _ { t } \big ) , } \\ { \displaystyle \delta \phi _ { m } \sim \mathcal { U } ( \mathcal { B } _ { r } ) , \mathrm { ~ f o r ~ } m = 1 , \dots , M . } \end{array}\tag{5}
$$

where $\mu _ { t } , \Sigma _ { t }$ are inferred from the Kalman predictor. Neural-Guided Parallel Refinement. Each hypothesis $\mathbf { T } _ { m }$ is then refined in parallel using a coarse-to-fine Levenberg-Marquardt (LM) optimizer over the geo-anchors,with a small number of iterations per pyramid level,as shown in Fig. 5(b-d). At each pyramid level l, we refine the pose hypothesis $\tilde { \mathbf { T } } _ { m }$ by minimizing the feature-based photometric cost $\mathcal { C } _ { \mathrm { p h o t o } } ^ { ( m , \ell ) }$ ,which measures the residual $r _ { j , \ell } ^ { ( m ) }$ between bilinearly sampled query features and reference features at a set of geo-anchors $P _ { j } ^ { \bar { W } }$

$$
\mathbf { r } _ { j , m } ^ { ( \ell ) } = \mathbf { f } _ { \ell } ^ { q } \left( \pi \left( \mathbf { K } _ { \ell } , \tilde { \mathbf { T } } _ { m } ^ { - 1 } , \mathbf { P } _ { j } ^ { W } \right) \right) - \mathbf { f } _ { \ell } ^ { r } ( p _ { j } ^ { r } )\tag{6}
$$

where $\pi ( \cdot )$ is the pinhole model. These per-anchor residuals are then aggregated into a cost function:

$$
\mathcal { C } _ { \mathrm { p h o t o } } ^ { ( m , \ell ) } = \sum _ { j } \rho \left( w _ { \ell } ( j ) \cdot \Vert \mathbf { r } _ { j , m } ^ { ( \ell ) } \Vert _ { 2 } ^ { 2 } \right)\tag{7}
$$

where $\rho ( \cdot )$ is the Huber robust loss [14],and $w _ { \ell } ( j )$ is a joint uncertainty score derived from $w _ { \ell } ^ { q }$ and $\boldsymbol { w } _ { \ell } ^ { r }$ at the corresponding pixel locations.

This update is performed iteratively for a total of K steps.At each iteration k, the pose is updated as:

$$
\left( \mathbf { J } ^ { \top } \mathbf { W } \mathbf { J } + \lambda \mathbf { I } \right) \Delta \pmb { \xi } = - \mathbf { J } ^ { \top } \mathbf { W } \mathbf { r } ,\tag{8}
$$

$$
\tilde { \mathbf { T } } _ { m } ^ { ( k + 1 ) } = \exp \left( \Delta \boldsymbol { \xi } \right) \cdot \tilde { \mathbf { T } } _ { m } ^ { ( k ) } .\tag{9}
$$

where $\Delta \xi \in \mathbb { R } ^ { 6 }$ is the Lie algebra increment, Jis the Jacobian of residuals r with respect to ε,and W is a diagonal matrix of uncertainty scores. Here, r,J,W are built at step $( m , \ell , k )$ The exponential map exp(·） denotes the SE(3) pose update.Further implementation details are provided in Sec.A.2 of the Appendix. After K iterations, the initial pose hypotheses are refined to $\{ \tilde { \mathbf { T } } _ { m } ^ { \prime } \}$

![](images/8a40e1ecd6c792f5f580795bff9519b29319371cf529df0a2442902bcdae7d7d.jpg)  
Figure 5.Rotation-aware sampling and coarse-to-fine optimization. The figure visualizes the pose convergence process in the pitch/yaw space: it synergizes wide-area Rotation-Aware Sampling(a) with parallel, coarse-to-fine refinement (b-d) to ensure robust convergence under aggressive motion.

Motion-Constrained Hypothesis Selection. To robustly select the best pose from multiple hypotheses,we additionally leverage the physics-based motion prior that favors poses registering with the predicted trajectory. We denote the KF predicted pose for the current frame as $\hat { \mathbf { T } } _ { \mathrm { p r e d } }$ Each hypothesis is scored by a total cost, which combines its final feature-based photometric cost $\mathcal { C } _ { \mathrm { p h o t o } } ^ { ( m , \ell = 2 ) }$ ,with a motion regularization term:

$$
\mathcal { C } _ { \mathrm { t o t a l } } ^ { ( m ) } = \mathcal { C } _ { \mathrm { p h o t o } } ^ { ( m , \ell = 2 ) } + \lambda \Vert \log ( \hat { \mathbf { T } } _ { \mathrm { p r e d } } ^ { - 1 } \tilde { \mathbf { T } } _ { m } ^ { ' } ) ^ { \vee } \Vert _ { 2 } ^ { 2 } .\tag{10}
$$

Here, $\mathcal { C } _ { \mathrm { m o t i o n } } ^ { ( m ) }$ computes the squared geodesic distance between the m-th hypothesis pose $\tilde { \mathbf { T } } _ { m } ^ { ' }$ and the KF-predicted pose $\hat { \mathbf { T } } _ { \mathrm { p r e d } }$ .This distance is formulated in the Lie algebra se(3)by mapping the relative transformation to its 6D twist vector representation using the $\log ( \cdot ) ^ { \vee }$ operation． The hyperparameter 入 balances the data and motion terms. The final pose T is determined by selecting the most reliable hypothesis, the one with the minimum total loss:

$$
m ^ { * } = \operatorname * { a r g m i n } _ { m } \mathcal { C } _ { \mathrm { t o t a l } } ^ { ( m ) } , \quad \hat { \mathbf { T } } = \tilde { \mathbf { T } } _ { m ^ { * } } ^ { ' } .\tag{11}
$$

Given ${ \hat { \mathbf { T } } } ,$ any query pixel can be mapped to geographic coordinates by casting a camera ray into M using the depth rendered from the estimated pose T.

## 4. Experiments

## 4.1. Experimental Setup

Implementation Details.We train our model using our proposed large-scale synthetic dataset (as detailed in Sec.3.3.1). This dataset provides training data in the form of reference-query pairs,sampled as consecutive frames along each generated UAV trajectory. We perturb query poses with random noise (5 \~ 15 m translation, $5 \sim 1 5 ^ { \circ }$ rotation） to simulate initialization uncertainty. Augmentations combine high-frequency (Fourier) noise [7] with photometric jitter (blur,contrast, Gaussian noise,brightness). Supervision is provided by a geometric reprojection loss on $N = 5 0 0$ reference geo-anchors. We train for 3O epochs using Adam $( \mathrm { l r } { = } 1 0 ^ { - 3 } )$ on 8 RTX 4090 GPUs.

At test time, we sample $M { = } 1 4 4$ pose hypotheses by pitch/yaw in $[ - 1 1 ^ { \circ } , 1 1 ^ { \circ } ]$ at $2 ^ { \circ }$ steps and add translation noise which obeys $\mathcal { N } ( 0 , 1 )$ m. Each hypotheses is optimized using 5OO sampling geo-anchors. We simulate a realistic initialization by assuming a coarse pose prior for the first frame with random noise of up to 1O m in translation and $1 0 ^ { \circ }$ in rotation, which is sourced from the ground truth of the first frame (for synthetic data) or coarse GNSS/IMU (for real-world data).

Baselines.We benchmark against methods in the context of map-based, scene-generalizable localization. The first category consists of hybrid methods that combine frameto-frame Visual Odometry (VO) with absolute pose corrections.We use ORB-SLAM3 (sparse feature-based) [5] and RAFT(optical flow-based) [3O] for relative tracking,periodically corrected at 1Hz by a Render-and-Compare module [38]. This module renders a reference view, finds correspondences with a LoFTR [29] matcher,and computes the absolute pose via PnP.We term these methods as Render2ORB and Render2RAFT.

The second category performs per-frame absolute localization,where each query frame is localized independentlyagainst a newlyrendered reference view.It includes PixLoc [27], which solves for the pose via direct alignment of dense feature maps,and Render2Loc [38],which instead employs a feature matching and PnP pipeline. For Render2Loc,we report results using four prominent matchers: LoFTR [29], EffcientLoFTR(ELoFTR) [35],and two recent SoTA matchers in aerial vision,Aerial-MASt3R [33] and RoMaV2 [11].

## 4.2. UAV-based Ego Localization

Datasets.We evaluate our method on three diverse benchmarks. The synthetic SynthCity-6 dataset, generated with Cesium for Unreal,covers 60 UAV trajectories (54k frames) over six 2km × 2km regions under various weather and lighting conditions.

We also test on the public UAVScenes [34] benchmark(51.6k frames from 01 runs across AMtown,AMvalley，HKairport,and HKisland scenes） and our newly built UAVD4L-2yr. UAVD4L-2yr pairs an outdated reference map with 8 new query flights (7.2k frames), introducing two-year seasonal and illumination gaps.It provides centimeter-level RTK-GPS ground truth for the UAV's 6- DoF pose and the corresponding 2D-3D annotations of dynamic targets within the scenes.Further dataset details are provided in Sec.B.2 of the Appendix.

Table 2. Comprehensive localization performance on synthetic and real-world datasets.We compare PiLoT against baselines across three diverse datasets,all using a shared map and 512 px inputs.Median errors are reported in meters (m) and degrees (°).
<table><tr><td></td><td>Medm↓</td><td colspan="4">SynthCity-6 (Synthetic)</td><td colspan="4">UAVScenes (Real)</td><td colspan="4">UAVD4L-2yr (Real)</td></tr><tr><td>Method</td><td>FPS↑</td><td></td><td>Med↓</td><td>R@1/3/5 (m,）↑</td><td>Comp.↑</td><td>Medm↓1</td><td>Med°↓</td><td>R@1/3/5(m,）↑</td><td>Comp.↑</td><td>Medm↓</td><td>Med°↓</td><td>R@1/3/5(m,）↑</td><td>Comp.↑</td></tr><tr><td>Hybrid Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Render2ORB</td><td>20.0</td><td>4.37</td><td>0.21</td><td>18.0 /30.2 /55.3</td><td>38.4</td><td>11.36</td><td>6.62</td><td>5.7 /63.5 /78.4</td><td>72.3</td><td>3.53</td><td>4.34</td><td>29.1/70.9 / 89.1</td><td>66.7</td></tr><tr><td>Render2RAFTt</td><td>10.0</td><td>5.21</td><td>0.62</td><td>8.4 /32.5 /48.9</td><td>96.2</td><td>10.54</td><td>4.21</td><td>4.2/45.8 / 62.3</td><td>88.5</td><td>2.68</td><td>2.51</td><td>22.4 / 62.6/77.4</td><td>92.3</td></tr><tr><td>Absolute Localizers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PixLoc†</td><td>5.0</td><td>0.55</td><td>0.06</td><td>65.4 / 92.3 /98.2</td><td>96.5</td><td>2.85</td><td>0.94</td><td>12.8 /94.6 /97.5</td><td>98.4</td><td>2.47</td><td>2.34</td><td>31.5 / 68.9 / 87.4</td><td>86.5</td></tr><tr><td>Render2Loc (LoFTR)†</td><td>2.0</td><td>0.49</td><td>0.04</td><td>76.5 /98.1/99.2</td><td>100.0</td><td>1.62</td><td>0.52</td><td>23.2/ 95.8 /98.9</td><td>100.0</td><td>1.08</td><td>0.95</td><td>44.2 /90.5 / 98.2</td><td>100.0</td></tr><tr><td>Render2Loc (ELoFTR)t</td><td>20.0</td><td>0.51</td><td>0.04</td><td>75.1/97.8 /99.0</td><td>100.0</td><td>1.84</td><td>0.65</td><td>20.6/95.2/98.4</td><td>100.0</td><td>1.15</td><td>1.08</td><td>44.0/ 86.8/97.7</td><td>98.7</td></tr><tr><td>Render2Loc (Aerial-MASt3R)*</td><td>0.4</td><td>0.52</td><td>0.05</td><td>69.8 /97.5 /99.1</td><td>100.0</td><td>1.85</td><td>0.72</td><td>21.2/94.8 /97.9</td><td>100.0</td><td>1.35</td><td>0.99</td><td>36.5 / 82.4 /93.1</td><td>100.0</td></tr><tr><td>Render2Loc (RoMaV2)*</td><td>0.8</td><td>0.47</td><td>0.04</td><td>77.2/98.8/99.6</td><td>100.0</td><td>1.42</td><td>0.58</td><td>22.1/95.5 /98.2</td><td>100.0</td><td>1.05</td><td>0.97</td><td>43.2/93.6/98.1</td><td>100.0</td></tr><tr><td>PiLoT (Ours)</td><td>28.0</td><td>0.46</td><td>0.03</td><td>80.4/99.9 /99.9</td><td>100.0</td><td>1.27</td><td>0.47</td><td>25.5/95.9/98.4</td><td>100.0</td><td>0.92</td><td>0.89</td><td>45.8/92.1/97.9</td><td>100.0</td></tr></table>

†Fine-tuned on our synthetic benchmark.\*SoTA matchers in aerial vision.

![](images/f9d69c7eb3a310faf1f2054d024692fe2aba772e498ea6fd8a13e73c1f85acfd.jpg)  
Figure 6.Per-frame stability on synthetic trajectories.Two trajectories with per-frame translation and rotation errors,showing that PiLoT(green） sustains a lower error distribution with fewer catastrophic outliers.

Metrics. We evaluate performance using four standard metrics:(1) Median Translation/Rotation Error (m,°） for accuracy on successfully localized frames; (2) Recall@1, 3, 5 (m,°),the percentage of frames localized within given translation and rotation thresholds; (3) Completeness (%), the percentage of frames for which a valid pose is produced without failure;and (4) Frequency (FPS), the rate of localization.

Results. The results are summarized in Tab.2.On the synthetic benchmark,PiLoT establishes a new state-of-theart, outperforming all baselines in localization accuracy and success rate while maintaining the fastest inference speed. Fig. 6 provides a quantitative, frame-by-frame analysis of localization stability under Foggy and Night conditions, where PiLoT yields more stable trajectories and lower pose errors.

On real-world data, deployed without any fine-tuning on the UAVScenes and UAVD4L-2yr, PiLoT again achieves state-of-the-art results with impressive zero-shot generalization.For a more in-depth analysis,we provide additional trajectory visualizations under varying motion speeds and weather conditions in the Appendix (see Figs.19 to 21).

## 4.3. UAV-based Target Geo-Localization

Datasets. We evaluate our method on two benchmarks:a synthetic UAVD4L-SynTarget and a real-world UAVD4L-2yr. UAVD4L-SynTarget is built with UE5 on the UAVD4L scene [36], including over 1OO synthetic targets (vehicles and pedestrians) across 6 UAV trajectories (6k frames) under diverse weather conditions. It provides precise ground truth for UAV poses, target 2D projections,and 3D geolocations.Further dataset generation details are provided in Sec.B.2 of the Appendix. UAVD4L-2yr is the same dataset used for ego-localization,but here we focus on its dynamic target tracking capabilities. Its eight trajectories include dynamic targets with precise annotations,making it ideal for this evaluation.

Metrics. We evaluate performance using Recall@k (m), defined as the percentage of targets geolocated within a 3D distance error of k meters from their ground truth position, where k ∈ {1,3,5}.

Results. We obtain target 3D coordinates via ray casting from the estimated UAV pose,assuming the target is a point-like object with a known 2D pixel location. As demonstrated in Tab.3,our method's localization translates into state-of-the-art performance, outperforming other methods across both synthetic and real-world scenarios. This performance gap is visualized qualitatively in Fig. 7. Our method's stable localization thus yields a consistently low-error target trajectory (uniform blue).

## 4.4. Framework Analysis

Ablation Study. Tab. 4 summarizes our quantitative findings. Our analysis confirms that domain-specific feature training is foundational.The effectiveness of our pro-posed training data is validated in the Training Data Ablation. Training on our synthetic dataset with diverse lighting and weather conditions significantly outperforms both realworld (MegaDepth [18]) and simpler synthetic baselines. The gain mainly arises from our dataset effectively bridging the domain gap and enabling feature adaptation to the top-down and oblique perspectives typical of UAV imagery.

![](images/3113bfbdba1e35a23d8dd6cd05472979b07068e3c6525ab6bd17bfd310088e6a.jpg)  
Figure 7. Qualitative results for joint UAV ego localization and dynamic target geo-localization on a UAVD4L-2yr sample sequence. (Top)Estimated UAV trajectories,where our method (PiLoT,green) most closely follows the ground truth (GT, black). (Middle） The position error of the dynamic target, with color indicating Euclidean distance to the GT.

Table 3. Performance on the dynamic target geo-localization task.PiLoT's superior ego-localization accuracy translates directly into state-of-the-art target geo-localization performance.We equip Render2loc with the LoFTR matcher,while PixLoc runs with its default 150 iterations.
<table><tr><td></td><td colspan="2">Dynamic Target Indication (R@1/3/5) 个</td></tr><tr><td>Method</td><td>Single-Target(Real)</td><td>Multi-Target(Syn)</td></tr><tr><td colspan="3">Hybrid Methods (Abs.+Rel.)</td></tr><tr><td>Render2ORB</td><td>72.13 / 84.59 / 89.74</td><td>79.51 /91.04 /93.28</td></tr><tr><td>Render2RAFT</td><td>44.15 /78.96/ 88.19</td><td>51.33 / 78.12 /90.50</td></tr><tr><td colspan="3">Absolute Localizers</td></tr><tr><td>PixLoc</td><td>83.37 /87.29 /91.85</td><td>86.15 /91.88 /93.91</td></tr><tr><td>Render2Loc</td><td>87.62/92.60 /96.25</td><td>89.03 /93.15 /96.07</td></tr><tr><td>PiLoT (Ours)</td><td>90.81/94.32/96.85</td><td>93.74/95.56/98.19</td></tr></table>

Table 4. Ablation study on system components and training data.We evaluate the contribution of each core component and the effectiveness of our proposed training dataset.
<table><tr><td></td><td colspan="3">Recall (%) @ 1m/1°↑</td></tr><tr><td>Ablation Configuration</td><td>w/3m,3</td><td>w/5m,5°</td><td>w/10m,10</td></tr><tr><td>System Components</td><td></td><td></td><td></td></tr><tr><td>Off-the-shelf Backbone</td><td>4.2</td><td>0.0</td><td>0.0</td></tr><tr><td>+ Domain-Specific Training</td><td>51.4</td><td>43.2</td><td>15.2</td></tr><tr><td>+ Rotation-Aware Hypothesis</td><td>83.8</td><td>78.9</td><td>70.6</td></tr><tr><td>+ Motion Regularizer (Ours)</td><td>84.3</td><td>84.3</td><td>84.2</td></tr><tr><td>Training Data Ablation</td><td></td><td></td><td></td></tr><tr><td>Trained on Syn. (no light/weather)</td><td>63.5</td><td>62.4</td><td>61.6</td></tr><tr><td>Trained on MegaDepth only</td><td>69.9</td><td>69.5</td><td>68.7</td></tr><tr><td>Trained on Ours (Syn.w/light/weather)</td><td>84.3</td><td>84.3</td><td>84.2</td></tr></table>

Building on this foundation,our rotation-aware hypothesis generation and motion regularizer further prove indispensable. Together, they substantially enhance the optimizer's robustness, enabling stable convergence even under aggressive flight maneuvers.

Runtime Efficiency.PiLoT achieves real-time performance via a dual-thread architecture and a CUDAaccelerated parallel optimizer. The former provides fresh reference views, while the latter embodies our core strategy of efficiently searching multiple pose hypotheses against a single reference image via massive parallelization. Detailed timing statistics are provided in Sec.A.3 of the Appendix. Limitations and Future Work.While PiLoT is robust to diverse environmental scenarios,its performance may degrade under extreme visual conditions (e.g.,dense fog) or significant calibration bias. Currently,our dependency on high-fidelity 3D models restricts the system's applicability to areas with pre-existing mesh data, hindering wider geographical expansion. To overcome these map acquisition constraints and broaden localization coverage,future work will focus on extending PiLoT to support universal representations,such as Digital Orthophoto Maps (DOM) and Digital Elevation Models (DEM). This transition wil facilitate seamless deployment across vast wilderness and urban environments.

## 5. Conclusion

This paper presents PiLoT,a unified framework for UAV ego and target geo-localization through direct registration of video streams to geo-referenced 3D maps.The proposed method introduces three key contributions:1) a dual-thread architecture that enables real-time performance while maintaining robustness,2）a large-scale synthetic dataset that facilitates zero-shot sim-to-real feature alignment,and 3) a joint neural-guided stochastic-gradient optimizer that ensures robust convergence under fast motion conditions.Extensive evaluations on both public benchmarks and a newly collected dataset demonstrate that PiLoT achieves state-ofthe-art accuracy while operating at over 25 FPS on embedded platforms.We believe this work not only advances the field of vision-based localization for UAVs under GNSSdenied conditions,but also provides valuable insights for localization in other robotic platforms.

## Acknowledgments

This research was funded through the Young Scientists Fund of the National Natural Science Foundation of China (NSFC) (Project No． 62406331). The authors would like to thank Rouwan Wu, Qing Shuai,Dongli Tan,and Na Zhao for their insightful discussions.We sincerely thank Cesium for Unreal for providing the data platform and Google Earth for providing the data source.

## References

[1]Relja Arandjelovic,Petr Gronat,Akihiko Torii, Tomas Pa-jdla,and Josef Sivic.Netvlad: CNN architecture for weakly supervised place recognition. In CVPR,2016.3

[2] Jonathan T Barron.A general and adaptive robust loss function. In CVPR,2019.5

[3] Gabriele Berton，Lorenz Junglas，Riccardo Zaccone, Thomas Pollok，Barbara Caputo，and Carlo Masone. Meshvpr:Citywide visual place recognition using 3d meshes.In ECCV,2024.3

[4] Yiming Cai, Yao Zhou, Hongwen Zhang,Yuli Xia, Peng Qiao,and Junsuo Zhao. Review of target geo-location algorithms for aerial remote sensing cameras without control points.Appl. Sci.,2022.3

[5] Carlos Campos,Richard Elvira, Juan J G6mez Rodriguez, José MMMontiel,and Juan D Tardós. Orb-slam3:An accurate open-source library for visual, visual-inertial,and multimap slam. IEEE T-RO,2021.2,6

[6] Shaozu Cao,Xiuyuan Lu,and Shaojie Shen. Gvins: Tightly coupled gnss-visual-inertial fusion for smooth and consistent state estimation.IEEE T-RO,2022.2

[7] Prithvijit Chattopadhyay，Kartik Sarangmath，Vivek Vi-jaykumar,and Judy Hoffman.Pasta: Proportional amplitude spectrum training augmentation for syn-to-real domain generalization. In ICCV,2023.6

[8] Ming Dai,Enhui Zheng,Zhenhua Feng,Lei Qi, Jiedong Zhuang， and Wankou Yang. Vision-based uavselfpositioning in low-altitude urban environments.IEEE TIP, 2023.3,4

[9] Daniel DeTone,Tomasz Malisiewicz,and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description.In CVPRW,2018.2, 3

[10] Oussema Dhaouadi,Riccardo Marin，Johannes Meier, Jacques Kaiser,and Daniel Cremers. Ortholoc: Uav 6-dof localization and calibration using orthographic geodata. arXiv preprint arXiv:2509.18350,2025.3

[11] Johan Edstedt,David Nordstrom,Yushan Zhang，Georg Bokman, Jonathan Astermark,Viktor Larsson,Anders Heyden,Fredrik Kahl,Marten Wadenbäck,and Michael Felsberg.Roma v2: Harder better faster denser feature matching. arXiv preprint arXiv:2511.15706,2025.6

[12] Stephen Hausler, Sourav Garg,Ming Xu,Michael Milford, and Tobias Fischer.Patch-netvlad: Multi-scale fusion of locally-global descriptors for place recognition．In CVPR, 2021.3

[13] Zhiwei Huang,Hailin Yu,Yichun Shentu,Jin Yuan,and Guofeng Zhang.From sparse to dense:Camera relocaliza-

tion with scene-specific detector from feature gaussian splatting.In CVPR,2025.3

[14] Peter JHuber.Robust estimation of a location parameter. In Breakthroughs in statistics:Methodology and distribution, 1992.5

[15] Yuxiang Ji,Boyong He,Zhuoyue Tan,and Liaoni Wu. Game4loc: A uav geo-localization benchmark from game data. In AAAI, 2025.3,4

[16] Bernhard Kerbl, Georgios Kopanas,Thomas Leimkuhler, and George Drettakis.3d gaussian splatting for real-time radiance field rendering.ACM TOG,2023.3

[17] Yixuan Li,Lihan Jiang,Linning Xu, Yuanbo Xiangli, Zhenzhi Wang,Dahua Lin,and Bo Dai. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In ICCV,2023.3,4

[18] Zhengqi Li and Noah Snavely. Megadepth: Learning singleview depth prediction from internet photos.In CVPR,2018. 8

[19] Tsung-Yi Lin,Yin Cui, Serge Belongie,and James Hays. Learning deep representations for ground-to-aerial geolocalization.In CVPR,2015.2

[20] Philipp Lindenberger,Paul-Edouard Sarlin,and Marc Pollefeys.Lightglue: Local feature matching at light speed. In ICCV,2023. 3

[21] Riku Murai，Eric Dexheimer,and Andrew J Davison. Mast3r-slam:Real-time dense slam with 3d reconstruction priors.In CVPR,2025.2

[22] Zhongyan Niu, Zhen Tan, Jinpu Zhang,Xueliang Yang,and Dewen Hu. Hgsloc:3dgs-based heuristic camera pose refinement.In ICRA,2025.3

[23] Vojtech Panek,Zuzana Kukelova,and Torsten Sattler. Meshloc:Mesh-based visual localization.In ECCV,2022. 2

[24] Tong Qin,Peiliang Li,and Shaojie Shen． Vins-mono:A robust and versatile monocular visual-inertial state estimator. IEEET-RO,2018.2

[25] Paul-Edouard Sarlin, Cesar Cadena,Roland Siegwart,and Marcin Dymczyk.From coarse to fine:Robust hierarchical localization at large scale.In CVPR,2019.2

[26] Paul-Edouard Sarlin,Daniel DeTone,Tomasz Malisiewicz, and Andrew Rabinovich.Superglue: Learning feature matching with graph neural networks.In CVPR,2020.3

[27] Paul-Edouard Sarlin,Ajaykumar Unagar,Mans Larsson, Hugo Germain,Carl Toft,Viktor Larsson,Marc Pollefeys, Vincent Lepetit,Lars Hammarstrand,Fredrik Kahl,et al. Back to the feature: Learning robust camera localization from pixels to pose.In CVPR,2021. 2, 3,6

[28] Paul-Edouard Sarlin,Daniel DeTone,Tsun-Yi Yang,Armen Avetisyan,Julian Straub,Tomasz Malisiewicz, Samuel Rota Bulo,Richard Newcombe,Peter Kontschieder,and Vasileios Balntas.Orienternet: Visual localization in 2d public maps with neural matching.In CVPR,2023.2

[29] Jiaming Sun, Zehong Shen, Yuang Wang,Hujun Bao,and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers.In CVPR,2021.2,3,6

[30] Zachary Teed and Jia Deng.Raft: Recurrent all-pairs field transforms for optical flow.In ECCV,2020.6

[31] Pavan Kumar Anasosalu Vasu,James Gabriel,Jeff Zhu, Oncel Tuzel,and Anurag Ranjan. Mobileone: An improved one millisecond mobile backbone.In CVPR,2023.11

[32] Nam N Vo and James Hays.Localizing and orienting street views using overhead imagery. In ECCV,2016.2

[33] Khiem Vuong,Anurag Ghosh,Deva Ramanan,Srinivasa Narasimhan,and Shubham Tulsiani.Aerialmegadepth: Learning aerial-ground reconstruction and view synthesis. In Proceedingsof the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21674-21684,2025.6

[34] Sijie Wang,Siqi Li, Yawei Zhang,Shangshu Yu, Shenghai Yuan,Rui She,Quanjiang Guo,JinXuan Zheng,Ong Kang Howe,Leonrich Chandra,et al.Uavscenes:A multi-modal dataset for uavs. In ICCV,2025.3,4,6

[35] Yifan Wang,XingyiHe, Sida Peng,Dongli Tan,and Xiaowei Zhou. Efcient loftr: Semi-dense local feature matching with sparse-like speed.In CVPR,2024.6

[36] Rouwan Wu,Xiaoya Cheng,Juelin Zhu, Yuxiang Liu,Mao-jun Zhang,and Shen Yan. Uavd4l: A large-scale dataset for uav 6-dof localization.In 3DV,2024.3,7,15

[37] Wenjia Xu, Yaxuan Yao,Jiaqi Cao,Zhiwei Wei, Chunbo Liu, Jiuniu Wang,and Mugen Peng.Uav-visloc:A large-scale dataset for uav visual localization.arXiv,2024.3

[38] Shen Yan,Xiaoya Cheng,Yuxiang Liu, Juelin Zhu,Rouwan Wu,Yu Liu,and Maojun Zhang.Render-and-compare: Cross-view 6-dof localization from noisy prior. In ICME, 2023. 2, 6

[39] Lin Yen-Chen,Pete Florence,Jonathan T Barron,Alberto Rodriguez,Philip Isola,and Tsung-Yi Lin.inerf: Inverting neural radiance fields for pose estimation.In IROS,2021.3

[40] Zhedong Zheng,Yunchao Wei,and Yi Yang．University-1652:A multi-view multi-source benchmark for dronebased geo-localization. In ACM MM,2020.3,4

[41] Qunjie Zhou, Torsten Sattler, and Laura Leal-Taixe. Patch2pix:Epipolar-guided pixel-level correspondences.In CVPR,2021.3

[42] Yu Zhou,Dong Tang,Hao Zhou,and Xin Xiang. Moving tar-get geolocation and trajectory prediction using a fixed-wing uav in cluttered environments. Remote Sens.,2025.3

[43] Runzhe Zhu,Ling Yin,Mingze Yang,Fei Wu,Yuncheng Yang,and Wenbo Hu. Sues-2OO: A multi-height multiscene cross-view image benchmark across drone and satellite.IEEETCSVT,2023.3,4

# PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization

Supplementary Material

## A.Method Details

## A.1. Network Architecture

Our feature extraction backbone is a lightweight U-Net architecture,with its detailed data flow outlined in Table 5. The encoder path employs the first three stages of a pretrained MobileOne-SO [31] to effciently extract a hierarchy of features at strides of 2,4,and 8.At each stage of the decoder, two parallel heads are applied to generate the multilevel outputs. A projection head, consisting of lightweight 3x3 convolutions,processes the decoder features to produce a feature map. Concurrently, an uncertainty head predicts a per-level single-channel uncertainty map.This entire process yields a three-level pyramid of feature-uncertainty pairs,indexed from coarse to fine $( \ell = 0 , 1 , 2 )$ corresponding to resolutions of $1 / 4 , 1 / 2$ ,and 1.

Table 5.Detailed data flow of our MobileOne-UNet feature extractor. The table illustrates the transformation of tensor shapes through the encoder,decoder,and output heads for an input image of resolution $H \times W = 5 1 2 \times 5 1 2$
<table><tr><td>Stage</td><td>Operation</td><td>Resolution</td><td>Channels</td><td>Description</td></tr><tr><td colspan="5">Encoder (MobileOne-S0)</td></tr><tr><td>Input</td><td></td><td>H×W</td><td>3</td><td>Input RGB image</td></tr><tr><td>Stage 1(E1)</td><td>Stem + Stage 1</td><td>(H/2)× (W/2)</td><td>C1</td><td>First level of features (for skip)</td></tr><tr><td>Stage 2(E2)</td><td>Stage 2 Blocks</td><td>(H/4)×(W/4)</td><td>C</td><td>Second level of features (for skip)</td></tr><tr><td>Stage 3(E3)</td><td>Stage 3 Blocks</td><td> $( H / 8 ) \times ( W / 8 )$ </td><td>C</td><td>Deepest features fed to decoder</td></tr><tr><td colspan="5">Decoder</td></tr><tr><td>Block 1(D1)</td><td>Upsample(E3)+ Concat(E2)</td><td> $( H / 4 ) \times ( W / 4 )$ </td><td>128</td><td>Upsample and fuse with skip connection</td></tr><tr><td>Block 2 (D2)</td><td>Upsample(D1) + Concat(E1)</td><td>(H/2)×(W/2)</td><td>64</td><td>Upsample and fuse with skip connection</td></tr><tr><td>Block 3 (D3)</td><td>Upsample(D2)</td><td>H×W</td><td>32</td><td>Final upsampling (no skip connection)</td></tr><tr><td colspan="5">OutputHeads (Applied toD,D,D3)</td></tr><tr><td rowspan="4">Projection</td><td>Conv 3x3</td><td> $( H / 4 ) \times ( W / 4 )$ </td><td>32</td><td>3-level feature pyramid</td></tr><tr><td></td><td>(H/2)× (W/2)</td><td>32</td><td></td></tr><tr><td></td><td>H×W</td><td>32</td><td></td></tr><tr><td>Conv lxl + Sigmoid</td><td></td><td>1</td><td></td></tr><tr><td rowspan="2">Uncertainty</td><td rowspan="2"></td><td> $\begin{array} { r } { \overline { { ( H / 4 ) \times ( W / 4 ) } } } \\ { \overline { { ( H / 2 ) \times ( W / 2 ) } } } \\ { H \times W } \end{array}$ </td><td>1</td><td rowspan="2">3-level confidence maps</td></tr><tr><td></td><td>1</td></tr></table>

## A.2. JNGO Optimizer Details

As illustrated in Fig. 8, the feature-metric cost landscape is often highly non-convex,posing a challenge for standard optimization methods. On one hand,exhaustive strategies like random sampling（Fig. 8 (a)) are too computationally expensive to be practical. On the other hand,eficient local search methods like gradient descent（Fig.8(b)） are highly sensitive to initialization and frequently get trapped in suboptimal local minima. To address this, JNGO synergizes stochastic and gradient-based optimization for effective global exploration and local refinement.

Iterative Linearization and Pose Update. The JNGO detailed here serves to minimize the feature-metric cost $\mathcal { C } _ { \mathrm { p h o t o } } ^ { ( m , \ell ) }$ for each hypothesis,as defined in Eq.6 of the main paper.

![](images/4c4470318a91043996f35dce5dc53605a21cb239240ffd45e6603eacecc60b78.jpg)  
Figure 8. Comparison of optimization strategies in a nonconvex landscape.While (a) Random Sampling is inefficient and (b) Gradient Descent is prone to local minima,our method (c) efficiently searches for the best solution. The convergence plot below illustrates how our method (green) achieves faster convergence and a lower final loss compared to alternatives.

We solve this non-linear optimization problem iteratively with a small, fixed number of updates per level to maintain real-time performance. Specifically, we perform 2, 3,and 4 LM iterations for the coarse,mid,and fine pyramid levels, respectively. While the full cost includes a robust Huber loss,the LM formulation solves the underlying non-linear least-squares problem based on the residual $\bar { \mathbf { r } } _ { j , \ell } ^ { ( m ) }$ from Eq.7.

Each LM iteration solves for a pose increment $\Delta \xi \ \in$ se(3)by linearizing the residual function. For a single residual term $\mathbf { r } _ { j }$ (where $j \in \{ 1 , \ldots , N \} )$ ，the linearization at $\tilde { \mathbf { T } } _ { m } ^ { ( k ) }$ is:

$$
\mathbf { r } _ { j } ( \exp ( \Delta \pmb { \xi } ) \cdot \tilde { \mathbf { T } } _ { m } ^ { ( k ) } ) \approx \mathbf { r } _ { j } ( \tilde { \mathbf { T } } _ { m } ^ { ( k ) } ) + \mathbf { J } _ { j } \Delta \pmb { \xi } ,\tag{12}
$$

where $\mathbf { J } _ { j }$ is the corresponding Jacobian． To solve for the update,we stack the residuals from all N 3D geo-anchors into a single vector $\mathbf { r } = [ \mathbf { r } _ { 1 } ^ { \top } , \ldots , \mathbf { r } _ { \mathbf { N } } ^ { \top } ] ^ { \top }$ and their Jacobians into a block matrix $\mathbf { J } = [ \mathbf { J } _ { 1 } ^ { \top } , \ldots , \mathbf { \bar { J } } _ { \mathbf { N } } ^ { \top } ] ^ { \top }$ ．This yields the normal equations:

$$
\left( \mathbf { J } ^ { \top } \mathbf { W } \mathbf { J } + \lambda \mathbf { I } \right) \Delta \pmb { \xi } = - \mathbf { J } ^ { \top } \mathbf { W } \mathbf { r } ,\tag{13}
$$

where W is a diagonal matrix of learned uncertainty weights.The resulting increment updates the pose $\tilde { \mathbf { T } } _ { m } ^ { ( k + 1 ) } = \exp ( \Delta \pmb { \xi } ) \cdot \tilde { \mathbf { T } } _ { m } ^ { ( k ) }$

Jacobian Formulation. The Jacobian matrix J encapsulates the sensitivity of the feature residual to infinitesimal pose perturbations and is derived via the chain rule for each

![](images/db35716a8d9d1d2679cb01724079d329c9ef9f548961a6522d23ece8a58fea83.jpg)  
Figure 9. Qualitative results of our multi-hypothesis refinement process.The top row shows a challenging rural scene with significant viewpoint change,while the bottom row demonstrates robustness in a day-night urban setting. The ‘convergence basin' visualization on the query image (right) demonstrates the subsequent refinement.

anchor point $j \colon$

$$
{ \bf J } _ { j } = \frac { \partial { \bf r } _ { j } } { \partial \pmb { \xi } } = \underbrace { \frac { \partial { \bf f } _ { \ell } ^ { q } ( \tilde { \bf p } _ { j } ^ { q } ) } { \partial \tilde { \bf p } _ { j } ^ { q } } } _ { \mathrm { F e a t u r e ~ G r a d i e n t } } \cdot \underbrace { \frac { \partial \pi ( { \bf P } _ { j } ^ { c } ) } { \partial { \bf P } _ { j } ^ { c } } } _ { \mathrm { P r o j e c t i o n ~ D e r i v . } } \cdot \underbrace { \frac { \partial ( \tilde { \bf T } _ { m } { \bf P } _ { j } ^ { W } ) } { \partial \pmb { \xi } } } _ { \mathrm { P o s e D e r i v . } } .\tag{14}
$$

The terms in this chain are:

·Feature Gradient: The $( C \times 2 )$ matrix $\frac { \partial \mathbf { f } _ { \ell } ^ { q } } { \partial \tilde { \mathbf { p } } _ { j } ^ { q } }$ represents the spatial gradient of the C-dimensional query feature map,typically computed using finite differences.

· Projection Derivative: The $( 2 \times 3 )$ matrix $\frac { \partial \pi } { \partial \mathbf { P } _ { j } ^ { c } }$ is the standard derivative of the pinhole camera projection with respect to the 3D point coordinates Pj in the camera frame.

· Pose Derivative: The (3 × 6) matrix $\frac { \partial ( \tilde { \mathbf { T } } _ { m } \mathbf { P } _ { j } ^ { W } ) } { \partial \pmb { \xi } }$ describes how the 3D point moves in the camera frame as a result of a pose perturbation.

The product of these terms yields the final $( C \times 6 )$ Jacobian block for a single anchor point,which links the 6-DoF pose update to changes in the feature residual. Figure 9 illustrates that the procedure begins by back-projecting 2D anchor points from the reference image to 3D,and then reprojecting them onto the query image based on initial pose hypotheses.For each seed point (colored dots in the reference image,left),our method initializes a local search region in the query image (right). The red arrows show the model predicting corrective pixel displacements,effectively converging towards the true location. For more convergence examples, please see Fig. 22.

Rotation-Aware Sampling Strategy. To justify our sampling design，we analyze the 6-DoF convergence basin of the proposed optimizer against the physical constraints of high-speed UAV motion.As reported in Table $^ { 6 , }$ we contrast our convergence limits with the maximum inter-frame motion (at 3O fps) derived from DJI Matrice 4 specifications. Our analysis reveals that while translation $( T _ { x } , T _ { y } , T _ { z } )$ and gimbal-stabilized Roll () axes stay well within the convergence bounds (marked in green), the outof-plane Yaw() and Pitch (0) motions frequently exceed the initial convergence limits $( 3 . 5 ^ { \circ } < 6 . 7 ^ { \circ }$ and $2 . 7 ^ { \circ } < 3 . 0 ^ { \circ }$ respectively).To bridge this gap and ensure robustness during aggressive maneuvers,we specifically expand the search space for Yaw and Pitch via the proposed Rotation-Aware Sampling strategy,effectively broadening the basin for these critical axes.

Algorithm1Fused CUDA Kernel fora SingleLMIteration   
1: Input: Pose hypotheses $\{ \tilde { \mathbf { T } } _ { m } ^ { ( k ) } \}$ , query features $\mathbf { f } ^ { q } ,$ an  
chor points $\{ \bar { \mathbf { P } } _ { j } ^ { W } \}$ ,weights $\{ w _ { j } \}$   
2: Output: System gradient g and Hessian H for each   
hypothesis   
3:for all hypothesis $\tilde { \mathbf { T } } _ { m } ^ { ( k ) }$ in parallel do   
4: Initialize global gradient $\mathbf { g } \in \mathbb { R } ^ { 6 }$ and Hessian H∈   
$\mathbb { R } ^ { 6 \times 6 }$ to zero.   
5: Launch one CUDA thread per anchor point $\mathbf { P } _ { j } ^ { W }$   
6: // --- In-thread computation   
(on-chip registers) ---   
7: Project point: $\mathbf { P } _ { j } ^ { c } = \tilde { \mathbf { T } } _ { m } ^ { ( k ) } \mathbf { P } _ { j } ^ { W }$ , then $\tilde { \mathbf { p } } _ { j } ^ { q } = \pi ( \mathbf { K } , \mathbf { P } _ { j } ^ { c } )$   
8: Compute residual $\mathbf { r } _ { j } = \mathbf { f } ^ { q } ( \mathbf { \bar { p } } _ { j } ^ { q } ) - \mathbf { f } ^ { r } ( \mathbf { \bar { p } } _ { j } ^ { r } )$   
9: Compute Jacobian $\mathbf { J } _ { j }$ via chain rule (Eq. (14))   
10: Compute per-point contribution: $\mathbf { g } _ { j } = \mathbf { J } _ { j } ^ { \top } w _ { j } \mathbf { r } _ { j }$ and   
$\mathbf { H } _ { j } = \mathbf { J } _ { j } ^ { \top } w _ { j } \mathbf { J } _ { j }$   
11: // --- Global memory reduction   
12: Update global system using at omicAdd:   
13: $\mathbf { g }  \mathbf { g } + \mathbf { g } _ { j }$   
14: H $ \mathbf { H } + \mathbf { H } _ { j }$   
15: end parallel for (threads sync implicitly)   
16: end for   
17:Solve (H + 入I)△ε = -g for all hypotheses on GPU.

Table 6. Comparison of 6-DoF convergence limits vs.maximum inter-frame motion (based on DJI specs at 30 fps).
<table><tr><td>Metric</td><td>Yaw(）</td><td>Pitch (0)</td><td>Roll()</td><td>Tx</td><td> $\mathbf { T y }$ </td><td>Tz</td></tr><tr><td>Conv. Limit</td><td> $3 . 5 ^ { \circ }$ </td><td> $2 . 7 ^ { \circ }$ </td><td> $3 . 2 ^ { \circ }$ </td><td>5.6m</td><td>8.1m</td><td>7.4m</td></tr><tr><td>Max Motion (30 fps)</td><td> $\cdot$ </td><td> $\cdot$ </td><td> $_ { \mathrm { i } } 0 . 1 ^ { \circ }$ </td><td>0.7m</td><td>0.7m</td><td>0.3m</td></tr></table>

Coarse-to-Fine Analysis.We further investigate the evolution of the convergence basin across different optimization scales.As illustrated in Fig.1O, the convergence basin exhibits a characteristic narrowing as the precision increases (Coarse → Mid →Fine).While the Fine level provides high-precision localization, its limited basin is often insufficient to capture large inter-frame displacements. Conversely, the Coarse level offers a significantly broader basin, providing the necessary robustness to large initial pose errors.This hierarchical behavior validates our coarse-to-fine design: the coarse stage provides a robust initialization that brings the pose within the tight convergence range of the fine stage,ultimately achieving a balance between global robustness and local precision.

![](images/87985c8bdb3a51d7bbe5741d5b0c0b739ae6f88aa510e86c7d4d6e03051c8f08.jpg)  
Figure 10. Convergence basin evolution across levels.

High-Performance CUDA Implementation with Kernel Fusion. To achieve real-time performance, the com-putationally intensive JNGO is accelerated with a custom CUDA implementation.A naive GPU port would involve multiple kernel launches for each step of the Jacobian calculation (e.g., projection, feature gradient sampling, matrix products),leading to significant memory bandwidth bottlenecks and launch overhead. Our key optimization is the use of a single, highly-fused CUDA kernel. As outlined in Al-gorithm 1, we parallelize the computation by launching one GPU thread for each of the Janchor points.Each thread is responsible for the entire calculation chain for its assigned point:

1. Projecting the 3D world point $\mathbf { P } _ { j } ^ { W }$ into the query camera frame.

2.CalculatingthefullJacobianchain $\begin{array} { r l } { \mathbf { J } _ { j } } & { { } = } \end{array}$ $\frac { \partial \mathbf { f } ^ { q } } { \partial \mathbf { p } ^ { q } } \frac { \partial \boldsymbol { \pi } } { \partial \mathbf { P } ^ { c } } \frac { \partial ( \mathbf { T } \mathbf { P } ^ { W } ) } { \partial \pmb { \xi } }$ and the feature residual $\mathbf { r } _ { j } . \quad \mathbf { A l l }$ intermediate" values are kept within fast on-chip registers.

3. Computing the per-point contribution to the final system: the gradient vector $\mathbf { g } _ { j } = \mathbf { J } _ { j } ^ { \top } \mathbf { W } _ { j } \mathbf { r } _ { j }$ and the Hessian matrix $\mathbf { H } _ { j } = \mathbf { J } _ { j } ^ { \top } \mathbf { W } _ { j } \mathbf { J } _ { j }$

4．Atomically adding these local contributions to the global gradient vector g and Hessian matrix H in global memory.

This strategy minimizes costly global memory access, maximizing computational throughput. Once the kernel completes,the final small $( 6 \times 6 )$ system of normal equations is solved in a batch for all M hypotheses using a parallel Cholesky decomposition on the GPU.As shown in Tab.7,our fused CUDA kernel dramatically reduces latency,achieving over 3Ox speedup,making our wide-area parallel search feasible in real-time.

Table7.Latency Comparison fora SingleLMIteration.We compare different implementations for computing the cost, gradient,and Hessian. Our fused CUDA kernel dramatically reduces latency,achieving over 3Ox speedup compared to the initial ONNXbased implementation.
<table><tr><td>Image Size</td><td>ONNX Runtime</td><td>C++ Reference</td><td>CUDA Port</td><td>CUDA Fused Kernel</td></tr><tr><td>512×512</td><td>14.9 ms</td><td>8.44 ms</td><td>5.72 ms</td><td>0.49 ms ★</td></tr><tr><td>256× 256</td><td>10.4 ms</td><td>8.35ms</td><td>5.24 ms</td><td>0.47 ms ★</td></tr><tr><td>128 ×128</td><td>8.8 ms</td><td>8.27 ms</td><td>5.21 ms</td><td>0.44 ms</td></tr></table>

## A.3.Dual-Thread Synchronization

Beyond low-level kernel optimization,PiLoT employs a synchronized dual-thread architecture to manage the interplay between map rendering and pose estimation. As detailed in Fig.11,our pipeline ensures strict temporal alignment: a new localization cycle only commences after the preceding rendering task is completed. This design maintains a consistent 2-frame lag between the reference view and the current query frame,which is essential for stable GPU memory management during high-speed rendering.

To mitigate the resulting latency (approx. 25 ms),we incorporate a constant-velocity motion model that extrapolates the UAV's pose to the exact rendering timestamp. Given our system operates at video-rate(< 4O ms intervals), the pose drift during this brief lag remains minimal.

<table><tr><td colspan="3">Map Rendering</td><td colspan="3">Feature Extraction Coarse LM(iter=4)</td><td colspan="2">Med LM(iter=3）</td><td colspan="2">Fine LM(iter=2)</td></tr><tr><td></td><td colspan="2">Frame i</td><td></td><td colspan="3">Frame i+1</td><td colspan="3">Frame i+2</td><td rowspan="2"></td></tr><tr><td colspan="3">Render Thread</td><td colspan="3">Render Ref.for Frame i+2 using pose i</td><td colspan="3"></td><td></td></tr><tr><td>Loc Thread</td><td colspan="2"></td><td>├Waiting -</td><td colspan="2"></td><td>-Waiting-</td><td></td><td>waiting-</td><td></td></tr><tr><td>0</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td></td><td>80Time(ms)</td></tr></table>

Figure 11. Timing diagram of the dual-thread synchronization.The pipeline maintains a fixed 2-frame lag,which is compensated by motion extrapolation to ensure real-time accuracy.

## B. Dataset Details

## B.1.Large-Scale Synthetic Dataset

In this section, we provide a detailed overview of our synthetic dataset, including the generation pipeline,quality validation procedures,and comprehensive statistics.

Data Generation Workflow. We developed a fully automated data acquisition pipeline,as shown in Fig. 12. This pipeline is engineered specifically to overcome the common geospatial and temporal incoherence in simulation-based data generation,ensuring high spatio-temporal coherence. The core components of our system include:

1. Geospatial Foundation: Leveraging Cesium for Unreal,which streams high-resolution photogrammetric 3D Tiles from Google, to provide a high-fidelity digital twin of the real world.

![](images/37e5388dd386e16c4b0af39717f9d59dd659b1bc32e18d9ef70edcb70c0652cf.jpg)  
Figure 12. The AirSim-Cesium-Unreal Engine Simulator Interface.Our system integrates Unreal Engine (UE) for real-time rendering,Cesium to load Google 3D Tiles models,and AirSim to simulate UAV flight missions. This pipeline enables the synchronous acquisition of photo-realistic query images and accurate 6-DoF ground truth poses within large-scale, geo-referenced environments.

2.Flight and Sensor Dynamics: Utilizing AirSim for simulating a multi-rotor UAV, enabling precise and reproducible trajectory execution based on WGS84 waypoints.

3.Photorealistic Rendering: Harnessing Unreal Engine's physically-based rendering engine to simulate a wide array of lighting and weather conditions.

Dataset Statistics and Diversity. Our dataset comprises over 1.1 million rendered images from 82 diverse regions, captured across over 65O km of UAV flight trajectories and featuring a rich mix of urban and natural landscapes. Environmental conditions are systematically varied across multiple weather conditions (Sunny, Cloudy, Rainy, Foggy, Snowy) and times of day (Day, Sunset, Night),and each region includes four trajectories rendered under distinct, randomly sampled weather settings to support cross-condition training and evaluation． The detailed distribution of these conditions and flight altitudes is summarized in Fig. 13. Our acquisition flights cover the sub-8OO m UAV operating envelope,and the camera pitch is swept from 2Oo(oblique)

![](images/9402b6fc6c10dec7155b2a9c04a34fcd21b626447b2749c152d94840d16d1ca7.jpg)

![](images/4459eacddc670eb1c5e722f13d486d8f3f9790636e7abeaef2ba3f8458f917c3.jpg)  
Figure 13.Dataset statistics for flight altitude (left) and environmental conditions with time of day (right).

![](images/fb93cd06b289bdbf41d1071003a95ef1c7424cbb1a688f28ba672364ea171099.jpg)

![](images/f4d7cf0ba9dc80591d507ab82a6483126c053dc257fa5049580ed86bbc34840b.jpg)  
Figure 14. Geometric Consistency Validation of Synthetic Data.Top:The Cumulative Distribution Function (CDF) of bidirectional relative depth errors across different altitudes (100m-800m).Bottom: Qualitative alignment results.The middle and bottom row illustrates the accurate mapping of co-visible regions across disparate perspectives.

to $9 0 ^ { \circ }$ (nadir) to approximate arbitrary viewpoints and en-able rigorous evaluation of robustness to viewpoint changes.

All images are rendered at a resolution of 16oO×1200. We use a pinhole camera model with per-frame intrinsics $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ For each frame,we provide the full 6-DoF pose, where the position is given in both WGS84 (longitude, latitude,height) and ECEF formats (X,Y, Z),and the rotation is provided as Euler angles (pitch,roll, yaw). This representation allows for seamless conversion to local project frames (e.g., for COLMAP or OSG) through well-defined transforms.

Data Quality and Validation. To ensure the geometric fidelity of our synthetic benchmark, we conduct both quantitative and qualitative validations. As illustrated in Fig.14, the Cumulative Distribution Function (CDF） of bidirectional relative depth errors across diverse altitudes (100m-800m） demonstrates high numerical precision，with over 90% of pixels exhibiting a relative depth error of less than 0.01 m. Qualitatively, depth-based warping results show seamless pixel-level alignment and consistent projection of co-visible regions across disparate viewpoints.

![](images/2c6994af05e24157573a5c5d52e2a6045261bd19772209cc034e41097456ddde.jpg)

Figure 15.Visualization of the designed "barrel-roll’ trajectory.The top-down view (top) shows yaw variation along the Sshaped path. The perspective views (bottom) illustrate changes in altitude (up-and-down arcs) and pitch (tilting of camera frustums).  
![](images/0e5ba3f84b36e1997e358a71b403d9b6b3ac2e5eb96102da98b840c8c43d9414.jpg)  
Figure 16. Data collection setup and trajectory visualization. (Left) The hardware used for data acquisition:a DJI M4T drone for capturing query images and Qianxun RTK instruments for ground truth positioning. (Right) A top-down view of the UAV flight path (query trajectory,shown in orange) and the corresponding ground truth (target trajectory, shown in purple).

Flight Trajectories Generation. We generate a set of “barrel-roll-inspired” style flight trajectories for data collection,as shown in Fig. 15. These trajectories combine horizontal orbiting around a central axis at a fixed radius with step-wise adjustments in pitch,yaw angle and altitude.The flight paths also integrate straight-line cruising and curved sweeps to introduce sufficient translational and viewpoint changes,providing the necessary data for training models to be robust against cross-view and cross-scale challenges.

## B.2. Evaluation Datasets

SynthCity-6 Dataset. We construct a synthetic test set, SynthCity-6,using the same data generation workflow as our training set.The test set uses six new locations from different regions of Switzerland and the USA,ensuring no geographic overlap with the training data. As detailed inTab.8，each sequence is rendered under

![](images/b326551233da96417947e6cb2695a34c8e2b1c369eb8e49913876d59ba4b653c.jpg)  
Figure 17. Sample checkerboard images used for intrinsic calibration.

5 weather/illumination conditions (sunny, sunset, night, foggy，cloudy/rainy）and at two different altitudes （200m and 5Oom) to introduce significant scale variations. In total, SynthCity-6 contains 54,OoO camera frames with synchronized 6-DoF poses, creating a comprehensive and challenging benchmark for assessing model robustness.For detailed trajectory visualizations and localization results on this dataset, please refer to Fig. 19 and Fig. 20.

UAVD4L-2yr Dataset.We build a new dataset for UAV-based ego and target geo-localization. For the reference map,we utilize the publicly available UAVD4L dataset [36]， which covers a large-scale urban area （100,OOO m²） containing diverse buildings，streets,and vegetation. Our query sequences are captured using a DJI Matrice 4T (M4T) drone 1. As detailed in Tab. 9,we collect data under varying illuminations (day and night), different environments (dense urban and sparse rural areas),and dynamic scenes with moving objects. Notably,a two-year time gap separates the query data from the reference map, posing a significant challenge due to long-term appearance variations. The flight paths and the target's trajectory are visualized in Fig.16.

The drone is equipped with a centimeter-level RTK module and a high-precision IMU to record its 6-DoF groundtruth pose. Concurrently, the ground-truth poses of the moving target were independently captured using a handheld Qianxun RTK device ².Both the drone's and the target's RTK systems were synchronized to a common UTC time source. All ground-truth poses were subsequently transformed into a unified ECEF coordinate system to en-sure precise spatio-temporal alignment between the query and target trajectories.The drone's video camera intrinsics were pre-calibrated using Zhang's method,as detailed in Fig.17. For detailed trajectory visualizations and localization results on this dataset, please refer to Fig. 20.

![](images/bde8ed8c869473fa5d93062d71662e81e99e6e9c01726577d1aa3ae3f5fe0703.jpg)

![](images/9ff6b944d881f6077dfd65d0692e3b5f38ffff6c30e1dbec323cdd0909d7699d.jpg)

![](images/857621a597150bd8c48ec31c1262e7a6b7465463ef680265cacd82a01eca1fd8.jpg)

![](images/ebb95ee73eaa2b212add5a3506943a6465a8406007c82a33fc4497fb5acd2269.jpg)

![](images/ff725af3da8d2cff965a5aa0243f2de4cb28c406bae655d43ff865f070082eef.jpg)  
Figure 18. Trajectory estimation results on Long Trajectory Flights. We compare our method's estimated trajectory (orange) against the ground truth from an RTK-IMU system (black). The plots show high consistency for planar position (XY),altitude, pitch,and yaw. The checkerboard insets provide an AR visualization,which overlays the live camera views with the rendered views based on our estimated pose.

![](images/38cb0a03a647717b3eb19a54ff31b75338e91e6012e88f4906663a900e55b61a.jpg)  
Figure 19. Trajectory estimation results on various synthetic scenes from the SynthCity-6 dataset. The figure showcases our method's performance across diverse synthetic conditions,including night (Switzerland-seq4,-seq7),cloudy (Switzerland-seql2),sunny (USA-seq2), and foggy (USA-seq5).

![](images/b59b7573d778dba735eae6bdaeabf81c4562e73ee898f434fe0f133050a84fe1.jpg)  
Figure 2o. Trajectory estimation results on challenging real-world and long-term scenarios. This figure demonstrates the model's robustness by evaluating on: (1) a synthetic sunset scene from SynthCity-6 (USA-seq8),and (2) 4 real-world sequences from the UAVD4L-2yr dataset, which include challenging day/night conditions (seq2,seq3,seq6,seq8).

![](images/41ef8ec21db5abcd466344129e376f19f74c600c3edfd391f643f7c5afc25c7c.jpg)

![](images/6a296e63d25867cfc09574b1333c56c05c8bbe7365d824157411d7499725b0f1.jpg)

![](images/536a0a404ea2786b5fdbe880dd72aa9deb478a8c4e19ae643487ca47ed4fb7c3.jpg)

![](images/c63ed3b1cd625abe2878f2e80d40a592b5ca834aac88162f3bd2daa0908b0c99.jpg)

![](images/53955a2b17a0d8cbf498de2db72ced882f0f73617538d88b91206ed6938f8ae4.jpg)

![](images/9ce8d7f634e19647e30c58985784bbb422484210397452bf3fd35d15f2670847.jpg)

![](images/29e5a2a96e5b96254b4c1b9acabbb51d942bab66bac0e4931fe717fedd981c1c.jpg)

![](images/1beda35a07b8cf5c865ab48aad867af856d38ccf338e9ccecbb22ebb7677d1f2.jpg)

![](images/878687ba474f7207d63aac7368cbaecaaf3b50a98d00e4a9478446a6ac3d2c81.jpg)

![](images/e9483f9905d50a3bbb088c813855b764fcedb96fda0e33d51d06b624703adb4d.jpg)

![](images/cb355a71800dd7fe3f1d89bdff67606198dbc4f04ce4f0f498eab410022ca36a.jpg)

![](images/35014aeec70a832995b5896617f776c9100df8822ef7cbabfd4a84b2d9de6b10.jpg)

![](images/e76b3451ef2633166288a5c3aaf72be6bc407437b5619c7d7b6677ca071fda99.jpg)

![](images/862df398839b9b1a79edd48552e54c51a94275d8c240208e25fc4695a8a40097.jpg)

![](images/169634bac849e6bbbbfcbb7c45aae78537dd53038d4bbbe3bad601eb78a82128.jpg)

![](images/4abe88421e9144e6b6cefeb05ef0c6353f3eb456a434d5a69918073ff4e75384.jpg)

![](images/f6ccd86cdbc0ae50b6e454b2cc198903c5a2ee860e663e0fa180252d17139f5c.jpg)  
Figure 21. Generalization performance on the standard UAVScenes benchmark dataset. This figure validates our model's strong generalization capability on four diverse,unseen scenes from the public UAVScenes benchmark． The scenes cover a wide range of environments:a town (AMtown),a natural valley (AMvalley),an airport (HKairport),and an island (HKisland).

Reference  
5 sample points on reference  
Convergence basin on query  
![](images/1e790b722f6abf734e064097cf1eb01f417fd940957bc58c0ce25285a0fa846b.jpg)  
Figure 22. Additional examples of the refinement process and its convergence basin. For various anchor points sampled in the reference image,the initial estimates in the query image (covering a wide area, shown as a field of arrows) are effectively guided by our learned features to converge to a single,precise location.

Table 8. Summary of SynthCity-6 dataset. Each trajectory provides camera frames with synchronized 6-DoF poses under varying weather, illumination,and altitude conditions.
<table><tr><td>Sequence</td><td>Weather /Illumination</td><td>Altitude (m)</td><td>Per Trajectory</td><td>Total Camera Frames</td><td>Location</td></tr><tr><td>Switzerland-seq4</td><td>Foggy/Night/Rainy/Sunny/Sunset</td><td>200/500</td><td>900</td><td>9000</td><td>Thun, Switzerland</td></tr><tr><td>Switzerland-seq7</td><td>Foggy/Night/Rainy/Sunny/Sunset</td><td>200/500</td><td>900</td><td>9000</td><td>Kirchlindach, Switzerland</td></tr><tr><td>Switzerland-seq12</td><td>Foggy/Night/Rainy/Sunny/Sunset</td><td>200/500</td><td>900</td><td>9000</td><td>Lausanne,Switzerland</td></tr><tr><td>USA-seq2</td><td>Foggy/Night/Cloudy/Sunny/Sunset</td><td>200/500</td><td>900</td><td>9000</td><td>Chicago,USA</td></tr><tr><td>USA-seq5</td><td>Foggy/Night/Cloudy/Sunny/Sunset</td><td>200/500</td><td>900</td><td>9000</td><td>New York,USA</td></tr><tr><td>USA-seq8</td><td>Foggy/Night/Cloudy/Sunny/Sunset</td><td>200/500</td><td>900</td><td>9000</td><td>California, USA</td></tr><tr><td>Total</td><td></td><td>1</td><td>1</td><td>54k</td><td>1</td></tr></table>

Table 9. Summary of UAVD4L-2yr dataset. Each sequence contains camera frames captured under varying illumination and altitude, covering different scene types with annotated primary target objects.
<table><tr><td>Sequence</td><td>Illumination</td><td>Altitude (m)</td><td>Camera Frames</td><td>Scene Type &amp; Environment</td><td>Target Object</td></tr><tr><td>UAVD4L-2yr-seq1</td><td>Daytime</td><td>70</td><td>900</td><td>suburban</td><td>dynamic person</td></tr><tr><td>UAVD4L-2yr-seq2</td><td>Daytime</td><td>70</td><td>900</td><td>urban</td><td>dynamic vehicles</td></tr><tr><td>UAVD4L-2yr-seq3</td><td>Daytime</td><td>70</td><td>900</td><td>urban</td><td>static landmark</td></tr><tr><td>UAVD4L-2yr-seq4</td><td>Daytime</td><td>110</td><td>900</td><td>suburban</td><td>dynamic vehicles</td></tr><tr><td>UAVD4L-2yr-seq5</td><td>Night</td><td>220</td><td>900</td><td>urban</td><td>dynamic vehicles</td></tr><tr><td>UAVD4L-2yr-seq6</td><td>Night</td><td>310</td><td>900</td><td>urban</td><td>dynamic vehicles</td></tr><tr><td>UAVD4L-2yr-seq7</td><td>Night</td><td>150</td><td>900</td><td>suburban</td><td>dynamic vehicles</td></tr><tr><td>UAVD4L-2yr-seq8</td><td>Night</td><td>150</td><td>900</td><td>urban</td><td>dynamic vehicles</td></tr><tr><td>Total</td><td>1</td><td>1</td><td>7.2k</td><td>1</td><td></td></tr></table>

Table 10. Summary of UAVD4L-SynTarget dataset. Each sequence contains camera frames captured under varying weather/illumination conditions and flight altitudes,with multiple target objects annotated along with their quantities.
<table><tr><td>Sequence</td><td>Weather/Illumination</td><td>Altitude (m)</td><td>Camera Frames</td><td>Target Object Number</td></tr><tr><td>UAVD4L-SynTarget-seq1</td><td>Sunny</td><td>130</td><td>304</td><td>&gt;100</td></tr><tr><td>UAVD4L-SynTarget-seq2</td><td>Sunny</td><td>130</td><td>344</td><td>&gt;100</td></tr><tr><td>UAVD4L-SynTarget-seq3</td><td>Sunny</td><td>180</td><td>1136</td><td>&gt;100</td></tr><tr><td>UAVD4L-SynTarget-seq4</td><td>Foggy</td><td>180</td><td>1136</td><td>&gt;100</td></tr><tr><td>UAVD4L-SynTarget-seq5</td><td>Night</td><td>180</td><td>1136</td><td>&gt;100</td></tr><tr><td>UAVD4L-SynTarget-seq6</td><td>Foggy</td><td>150</td><td>1192</td><td>&gt;100</td></tr><tr><td>UAVD4L-SynTarget-seq7</td><td>Sunny</td><td>240</td><td>808</td><td>&gt;100</td></tr><tr><td>Total</td><td>1</td><td>1</td><td>6k</td><td>1</td></tr></table>