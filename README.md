# Code
```
# Ignore warning python
import warnings
warnings.filterwarnings("ignore")

# Scroll dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Matplotlib / seaborn background fix
sns.set(style="white", font_scale=1.0, font="MS Gothic")

# List of files in subfolder
list_path, , accepted = '', [], ['.jpeg','.png']
for root, dirs, files in tqdm(os.walk(data_dir)):
    for file in files:
        if os.path.splitext(file)[1].lower() in accepted:
            list_path.append(os.path.join(root, file))
# Sort list of tuple
lst = sorted(lst, key=lambda x: x[1])
```

# Notebooks -  Various basic operations in notebook format
- [Hyperparameters optimization - gridsearch & bayesian using hyperopt and sklearn](./notebooks/hyperparameter_optimization.ipynb)
- [XGB multiple output](./notebooks/multiple_output_xgboost.ipynb)
- [Image encode/decode base64](./notebooks/base_64_image_conversion.ipynb)
- [Video creation (random noise)](./notebooks/random_noise_video.ipynb)
- [Random color generation](./notebooks/random_color.ipynb)
- [DBSCAN / kmean clustering (elbow/silhouette)](./notebooks/dbscan_kmeans_clustering.ipynb)
- [Optimal skew processing](./notebooks/unskew_data_distribution.ipynb)
- [Over sampling, under sampling](./notebooks/undersampling_oversampling.ipynb)
- [Logging](./notebooks/logging.ipynb)
- [EDA & CFA](./notebooks/eda_cfa.ipynb)
- [Fastest strin count in list benchmark](./notebooks/benchmark_string_count.ipynb)
- [Webcam stream acquisition](./notebooks/capture_webcam_notebook.ipynb)
- [Keras models to tflite](./notebooks/keras_to_tflite.ipynb)

# Notes about AI in general

## Various
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2022/01/05 | [DL and gaming](https://www.reddit.com/r/MachineLearning/comments/rw50hg/d_deep_learning_is_the_future_of_gaming/)                                                    |[here](./notes/DL_and_gaming.md)                   |                   |
| 2021/10/01 | [Limits of CNNs progresses](https://www.reddit.com/r/MachineLearning/comments/pyti87/d_have_we_reached_some_limit_in_the_advancement/)                                |[here](./notes/limits_of_CNNs.md)                  |                   |
| 2021/04/13 | [Tesla autopilot](https://www.reddit.com/r/MachineLearning/comments/mq3led/d_how_is_tesla_autopilot_trained/)                                                         |[here](./notes/tesla_autopilot.md)                 |                   |
| 2021/12/28 | [New AI approaches](https://www.reddit.com/r/MachineLearning/comments/rq6uih/d_other_ai_methodsalgorithms_except_deep_neural/)                                        |[here](./notes/new_algorithms.md)                  |                   |
| 2021/12/16 | [Search engine for repositories](https://www.reddit.com/r/learnmachinelearning/comments/rh25sh/ive_made_a_search_engine_with_5000_quality_data/)                      |[here](./notes/repo_search_engine.md)              |                   |
| 2021/11/24 | [Huggingface performance optimization](https://www.reddit.com/r/MachineLearning/comments/r0y56t/p_python_library_to_optimize_hugging_face/)                           |[here](./notes/huggingface_performances.md)        |                   |
| 2021/09/30 | [Less recommended cases for transformers](https://www.reddit.com/r/MachineLearning/comments/pxz1iw/d_nlp_or_computer_vision_tasks_where_transformers/)                |[here](./notes/no_transformers_use_cases.md)       |                   |
| 2021/09/25 | [Feature matching with transformers](https://www.reddit.com/r/MachineLearning/comments/puz9kw/r_loftr_detectorfree_local_feature_matching_with/)                      |[here](./notes/loftr_feature_matching.md)          |                   |
| 2021/04/23 | [Easy model finetuning library](https://www.reddit.com/r/MachineLearning/comments/mas3sz/p_backprop_a_library_to_easily_finetune_and_use/)                            |[here](./notes/finetune_backprop.md)               |                   |
| 2021/03/11 | [Cloud solutions](https://www.reddit.com/r/MachineLearning/comments/m1zrxy/d_google_cloud_vs_other_cloud_solutions_for/)                                              |[here](./notes/cloud_solutions.md)                 |                   |
| 2021/01/15 | [SHAPash - Model explanability](https://www.reddit.com/r/MachineLearning/comments/kxrld8/p_introducing_shapash_a_new_python_library_makes/)                           |[here](./notes/shapash_explanability.md)           |                   |
| 2020/11/21 | [Keypoints modelling](https://www.reddit.com/r/MachineLearning/comments/qymvys/r_rethinking_keypoint_representations_modeling/)                                       |                                                   |                   |
| 2020/09/15 | [Newest algorithm developed by FAANG](https://www.reddit.com/r/MachineLearning/comments/it44ix/r_new_ml_algorithms_developed_by_facebook/)                            |[here](./notes/faang_algorithms.md)                |                   |
| 2020/08/20 | [Pretrained models commercial use law](https://www.reddit.com/r/MachineLearning/comments/id4394/d_is_it_legal_to_use_models_pretrained_on/)                           |[here](./notes/pretrained_models_law.md)           |                   |
| 2020/07/09 | [GridCSearchCV - hyperparameters optimization](https://www.reddit.com/r/MachineLearning/comments/hnn1vv/p_gridsearchcv_20_up_to_10x_faster_than_sklearn/)             |[here](./notes/tunesklearn.md)                     |                   |
| 2020/04/25 | [Streamer background](https://www.reddit.com/r/MachineLearning/comments/g7ntb8/r_background_matting_the_world_is_your_green/)                                         |[here](./notes/green_screen.md)                    |                   |
| 2020/01/20 | [Relative's voice](https://www.reddit.com/r/MachineLearning/comments/er3ng8/d_how_to_save_my_fathers_voice/)                                                          |[here](./notes/voice_saving.md)                    |                   |
| 2020/01/08 | [Henri AI youtube channel](https://www.reddit.com/r/MachineLearning/comments/elt7p6/n_henry_ai_labs_on_youtube/)                                                      | Various well illustrated AI concepts              |                   |
| 2019/12/26 | [AI engineers frustration](https://www.reddit.com/r/MachineLearning/comments/eftv1o/d_what_frustrates_you_about_ml_tools_libraries/)                                  |                                                   |                   |
| 2019/12/25 | [DL definition by Lecun](https://www.reddit.com/r/MachineLearning/comments/ef7cbb/d_yann_lecun_some_folks_still_seem_confused_about/)                                 |                                                   |                   |
| 2019/12/29 | [Open source project 2019 v0](https://www.reddit.com/r/MachineLearning/comments/egyp7w/d_what_is_your_favorite_opensource_project_of/)                                |                                                   |                   |
| 2019/12/13 | [Open source projects 2019 v1](https://www.reddit.com/r/MachineLearning/comments/e9rwj9/d_what_do_you_think_were_the_most_important_open/)                            |                                                   |                   |
| 2019/11/18 | [Hyperparameters optimisation in research papers](https://www.reddit.com/r/MachineLearning/comments/dy23rm/d_many_papers_dont_do_hyperparameter_search_on/)           |                                                   |                   |


## Image
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2021/11/25 | [Attention mechanism in video understanding](https://www.reddit.com/r/MachineLearning/comments/qeyhwb/r_efficient_visual_selfattention_link_to_a_free/)               |                                                   |                   |
| 2021/10/09 | [Pose estimation](https://www.reddit.com/r/MachineLearning/comments/q4eicp/r_keypoint_communities/)                                                                   |                                                   |                   |
| 2021/09/21 | [GAN for unfamiliar scene](https://www.reddit.com/r/MachineLearning/comments/ps5ubp/r_facebook_ai_introduces_a_new_image_generation/)                                 |                                                   |                   |
| 2021/09/21 | [CV as inverse computer graphics - Mesh](https://www.reddit.com/r/MachineLearning/comments/pryveo/d_computer_vision_as_inverse_computer_graphics/)                    |                                                   |                   |
| 2021/09/12 | [STRIVE - Text replacement in image](https://www.reddit.com/r/MachineLearning/comments/pmn8nq/r_ai_researchers_from_amazon_nec_stanford_unveil/)                      |                                                   |                   |
| 2021/06/06 | [Line retrieval](https://www.reddit.com/r/MachineLearning/comments/nt8hlp/p_towards_realtime_and_lightweight_line_segment/)                                           |                                                   |                   |
| 2021/04/19 | [Unusual scene object detection](https://www.reddit.com/r/MachineLearning/comments/mtev6w/r_putting_visual_recognition_in_context_link_to/)                           |                                                   |                   |
| 2021/04/03 | [Image editing with text](https://www.reddit.com/r/MachineLearning/comments/minzbz/r_styleclip_textdriven_manipulation_of_stylegan/)                                  |                                                   |                   |
| 2021/02/11 | [Transformers in CV](https://www.reddit.com/r/MachineLearning/comments/lh7iwp/d_why_did_it_took_3_years_to_use_transformers_in/)                                      |                                                   |                   |
| 2021/01/17 | [Notes digitization](https://www.reddit.com/r/MachineLearning/comments/kykhc1/p_digitize_your_notes/)                                                                 |                                                   |                   |
| 2020/08/28 | [Image to 3D pose and mesh](https://www.reddit.com/r/MachineLearning/comments/iej5cb/news_heres_a_new_paper_announced_in_the_eccv2020/)                               |                                                   |                   |
| 2020/07/08 | [SCAN - Classify without labels](https://www.reddit.com/r/MachineLearning/comments/hni969/research_official_pytorch_implementation_for_scan/)                         |                                                   |                   |
| 2020/04/27 | [StarGAN v2](https://www.reddit.com/r/MachineLearning/comments/g8s1af/r_clova_ai_researchs_stargan_v2_cvpr_2020_code/)                                                |                                                   |                   |
| 2020/04/18 | [GAN in CV](https://www.reddit.com/r/MachineLearning/comments/g3lh3n/d_gans_in_computer_vision_an_article_review/)                                                    |                                                   |                   |
| 2019/12/22 | [Monocular depth](https://www.reddit.com/r/MachineLearning/comments/ee2khy/d_monocular_depth_perception_of_autonomous/)                                               |                                                   |                   |


## Text
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2020/08/19 | [Sentiment analysis in 2020](https://www.reddit.com/r/MachineLearning/comments/ic5nzp/d_what_is_the_best_way_for_sentiment_analysis_in/)                              |                                                   |                   |
| 2020/06/03 | [NLP colab](https://www.reddit.com/r/MachineLearning/comments/gvsh51/p_181_nlp_colab_notebooks_found_here/)                                                           |                                                   |                   |
| 2020/04/30 | [Text classification tips and tricks](https://www.reddit.com/r/MachineLearning/comments/gaqm5z/d_list_of_text_classification_tips_and_tricks/)                        |                                                   |                   |
| 2019/11/27 | [Single Headed Attention RNN](https://www.reddit.com/r/MachineLearning/comments/e2ch4t/r_single_headed_attention_rnn_stop_thinking_with/)                             |                                                   |                   |


## Tabular
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2021/09/27 | [Graph NN for pointcloud](https://www.reddit.com/r/MachineLearning/comments/pvyvet/r_graph_neural_networks_for_point_cloud_processing/)                               |                                                   |                   |
| 2021/07/20 | [Sparse high dimensional data](https://www.reddit.com/r/MachineLearning/comments/onyofp/d_what_is_the_method_to_deal_with_sparse_high/)                               |                                                   |                   |
| 2019/11/22 | [Hierarchical Bayesian regression on imbalanced data](https://www.reddit.com/r/MachineLearning/comments/dzmssp/d_why_does_hierarchical_bayesian_regression_work/)     |                                                   |                   |

## Notebooks
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2021/12/18 | [PowerBI in jupyter](https://www.reddit.com/r/MachineLearning/comments/rimqij/d_how_to_embed_powerbi_report_in_jupyternotebook/)                                      |[here](./notes/jupyter_powerbi.md)                 |                   |
| 2021/10/07 | [Notebook to production](https://www.reddit.com/r/MachineLearning/comments/q344pp/notebook_to_production_d/)                                                          |[here](./notes/notebook_to_production.md)          |                   |
| 2020/12/10 | [Share notebook results easily](https://www.reddit.com/r/MachineLearning/comments/k9xuri/p_chrome_extension_to_share_your_results_from/)                              |[here](./notes/share_results_notebook.md)          |                   |
| 2020/06/07 | [Pandas read plot](https://www.reddit.com/r/MachineLearning/comments/gxuz7y/d_organize_your_pandas_notebook_with_a_cool_hack/)                                        |[here](./notes/pandas_plot.md)                     |                   |

## Recommendation system
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2020/02/13 | [Tiktok for you page](https://www.reddit.com/r/MachineLearning/comments/f38hxi/d_how_does_tiktok_manage_to_optimize_the_for_you/)                                     |                                                   |                   |
| 2019/11/29 | [From feature map to recommendation](https://www.reddit.com/r/MachineLearning/comments/e3ctv5/d_helpquestion_about_using_vector_projection/)                          |                                                   |                   |


## Data
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2021/12/02 | [Large XML to dataframe](https://www.reddit.com/r/learnmachinelearning/comments/r6jee3/how_to_load_856_gb_of_xml_data_into_a_dataframe/)                              |[here](./notes/large_xml_df.md)                    |                   |
| 2021/11/21 | [French NLP dataset - Cedille](https://www.reddit.com/r/MachineLearning/comments/qqzuh0/p_cedille_the_largest_french_language_model_6b/)                              |[here](./notes/cedille_french_nlp.md)              |                   |
| 2021/11/19 | [Bias in data](https://www.reddit.com/r/MachineLearning/comments/qx0enm/d_all_bias_in_ml_comes_from_biased_data/)                                                     |                                                   |                   |
| 2021/06/05 | [HDF5 for data storage](https://www.reddit.com/r/MachineLearning/comments/nsq3ai/p_h5records_store_large_datasets_in_one_single/)                                     |[here](./notes/hdf5.md)                            |                   |
| 2020/07/18 | [Image segmentation data exploration](https://www.reddit.com/r/MachineLearning/comments/hbb8qm/d_data_exploration_for_image_segmentation_and/)                        |                                                   |                   |
| 2020/05/12 | [Dataset similarity comparison](https://www.reddit.com/r/datasets/comments/gi6282/datagene_a_python_package_to_identify_how_similar/)                                 |                                                   |                   |
| 2020/05/04 | [400 NLP datasets](https://www.reddit.com/r/MachineLearning/comments/gdbz0r/p_400_nlp_datasets_found_here/)                                                           |                                                   |                   |
| 2019/28/11 | [COCO face dataset](https://www.reddit.com/r/MachineLearning/comments/e2pi0q/project_faces4coco_dataset_released_face_bounding/)                                      |                                                   |                   |
| 2015/07/03 | [Reddit comments dataset](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/)                                          |                                                   |                   |


## Theoretical
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2021/10/26 | [Kernel eigenvalues to predict generalization](https://www.reddit.com/r/MachineLearning/comments/qfy76l/r_neural_tangent_kernel_eigenvalues_accurately/)              |                                                   |                   |
| 2021/10/16 | [ResNet feature map size](https://www.reddit.com/r/MachineLearning/comments/q8qfs2/discussion_what_is_the_best_way_to_extract/)                                       |                                                   |                   |
| 2020/07/30 | [GAN convergence trick](https://www.reddit.com/r/MachineLearning/comments/i085a8/d_best_gan_tricks/)                                                                  |                                                   |                   |
| 2020/07/05 | [Perceptual loss for image](https://www.reddit.com/r/MachineLearning/comments/hlkxds/d_vgg_perceptual_loss_for_grayscale_images/)                                     |                                                   |                   |
| 2019/12/30 | [GELU activation](https://www.reddit.com/r/MachineLearning/comments/eh80jp/d_gelu_better_than_relu/)                                                                  |                                                   |                   |
| 2019/12/26 | [ResNet performance on MNIST](https://www.reddit.com/r/MachineLearning/comments/eft2bs/resnet_and_mnist_r/)                                                           |                                                   |                   |
| 2019/12/24 | [Autoencoder symmetry](https://www.reddit.com/r/MachineLearning/comments/ef1xe8/d_should_autoencoders_really_be_symmetric/)                                           |                                                   |                   |
| 2019/12/23 | [Constrained optimization](https://www.reddit.com/r/MachineLearning/comments/eeirql/d_i_want_to_optimize_my_model_based_on_two/)                                      |                                                   |                   |
| 2019/12/23 | [lr / gradient clipping](https://www.reddit.com/r/MachineLearning/comments/eea88q/d_relationship_between_learning_rate_and_gradient/)                                 |                                                   |                   |
| 2019/12/01 | [New basic operators](https://www.reddit.com/r/MachineLearning/comments/e3ykhf/d_nas_has_anyone_tried_yet_to_search_for_new/)                                         |                                                   |                   |
| 2019/11/27 | [Bacth size impact on generalization](https://www.reddit.com/r/MachineLearning/comments/e2afb4/d_what_is_the_latest_consensus_on_the_effect_of/)                      |                                                   |                   |


## Publications / blogs
|    Date    | Source                                                                                                                                                                |                   Detailed note                   | Notes / conlusion |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2021/12/29 | [Graph neural network](https://www.reddit.com/r/learnmachinelearning/comments/rqukh3/what_are_graph_neural_networks/)                                                 |[here](./notes/graph_neural_network.md)            |                   |
| 2021/11/18 | [Bert for image](https://www.reddit.com/r/MachineLearning/comments/qw2c3p/r_is_bert_the_future_of_image_pretraining)                                                  |                                                   |                   |
| 2021/10/14 | [Limits of large-scale model pretraining](https://www.reddit.com/r/MachineLearning/comments/q81eax/r_google_researchers_explore_the_limits_of/)                       |                                                   |                   |
| 2021/10/13 | [Model parameters versus number of label](https://www.reddit.com/r/MachineLearning/comments/q739y5/r_a_few_more_examples_may_be_worth_billions_of/)                   |                                                   |                   |
| 2021/10/06 | [SotA GAN based image editing](https://www.reddit.com/r/MachineLearning/comments/q1z4hg/d_sota_ganbased_image_editing_isfgan_an_implicit/)                            |                                                   |                   |
| 2021/10/04 | [ResNet strikes back](https://www.reddit.com/r/MachineLearning/comments/q0vt2b/r_resnet_strikes_back_an_improved_training/)                                           |                                                   |                   |
| 2021/09/02 | [Lesson from AI related publications](https://www.reddit.com/r/MachineLearning/comments/pgitms/d_here_is_what_i_learned_from_writing_50/)                             |                                                   |                   |
| 2021/06/22 | [DL regularization techniques for table data](https://www.reddit.com/r/MachineLearning/comments/o5nmoz/r_regularization_is_all_you_need_simple_neural/)               |                                                   |                   |
| 2021/04/24 | [BEV object path prediction](https://www.reddit.com/r/MachineLearning/comments/mx1t3v/r_fiery_future_instance_prediction_in_birdseye/)                                |                                                   |                   |
| 2021/01/21 | [Community best research papers](https://www.reddit.com/r/MachineLearning/comments/l1gyp6/r_what_are_some_of_the_best_research_papers_to/)                            |                                                   |                   |
| 2020/12/07 | [ADAM as best optimizer](https://www.reddit.com/r/MachineLearning/comments/k7yn1k/d_neural_networks_maybe_evolved_to_make_adam_the/)                                  |                                                   |                   |
| 2020/06/09 | [Unsupervised translation of programming languages](https://www.reddit.com/r/MachineLearning/comments/gz9pcx/r_unsupervised_translation_of_programming/)              |                                                   |                   |
| 2020/05/12 | [Group normalization](https://www.reddit.com/r/MachineLearning/comments/gibvs8/d_paper_explained_group_normalization/)                                                |                                                   |                   |
| 2020/05/05 | [Image Super-Resolution via GAN](https://www.reddit.com/r/MachineLearning/comments/gdt35p/d_unsupervised_real_image_superresolution_via/)                             |                                                   |                   |
| 2020/04/29 | [Image Augmentation Is All You Need](https://www.reddit.com/r/MachineLearning/comments/ga19q8/r_image_augmentation_is_all_you_need_regularizing/)                     |                                                   |                   |
| 2020/02/27 | [Generation of vertex-by-vertex meshes using Transformers](https://www.reddit.com/r/MachineLearning/comments/f9uryf/r_polygen_an_autoregressive_generative_model_of/) |                                                   |                   |
| 2020/01/07 | [Lipschitz regularization vs gradient penalty](https://www.reddit.com/r/MachineLearning/comments/el9cq9/r_adversarial_lipschitz_regularization/)                      |                                                   |                   |
| 2019/12/22 | [Conditional Image Generation](https://www.reddit.com/r/MachineLearning/comments/ee5fab/mixnmatch_multifactor_disentanglement_and/)                                   |                                                   |                   |
| 2019/12/24 | [Best papers nips_acl_emnlp_2019](https://www.reddit.com/r/MachineLearning/comments/eep8yp/d_summary_of_best_papers_of_nips_acl_emnlp_2019/)                          |                                                   |                   |
| 2019/12/01 | [Batch dependence](https://www.reddit.com/r/MachineLearning/comments/e4g50h/r_filter_response_normalization_layer_eliminating/)                                       |                                                   |                   |
| 2019/11/30 | [PhD student paper recap](https://www.reddit.com/r/MachineLearning/comments/e3fwat/d_ml_paper_notes_my_notes_of_various_ml_research/)                                 |                                                   |                   |
| 2019/11/15 | [MICCAI 2019 papers](https://www.reddit.com/r/MachineLearning/comments/dwbcxy/n_awesome_ai_research_and_papers_reviewed_on/)                                          |                                                   |                   |


## To be investigated
|    Date    | Source                                                                                                                                                     |                   Detailed note                   | Notes / conlusion |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------------------|
| 2020/01/01 | [selfsupervised_3d](https://www.reddit.com/r/MachineLearning/comments/gz1gpg/p_selfsupervised_3d_keypoint_learning_for/)                                   |                                                   |                   |
| 2020/01/01 | [reinforcement_learning](https://www.reddit.com/r/MachineLearning/comments/gpmbpl/projectreinforcement_learning_using_dqn_qlearning/)                      |                                                   |                   |
| 2020/01/01 | [two_soccer](https://www.reddit.com/r/MachineLearning/comments/g7tzxd/p_training_twoontwo_soccer_agents_using_selfplay/)                                   |                                                   |                   |
| 2020/01/01 | [objective_masked_language](https://www.reddit.com/r/MachineLearning/comments/eelbd6/d_objective_masked_language_model_vs_autoencoding/)                   |                                                   |                   |
| 2020/01/01 | [implementing_ambient_sound](https://www.reddit.com/r/MachineLearning/comments/eefml7/p_implementing_ambient_sound_provides_supervision/)                  |                                                   |                   |
