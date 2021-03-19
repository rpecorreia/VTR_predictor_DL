# VTR predictor with deep learning

Online Advertising uses different KPIs (Key Performance Indicators) to assess the quality of advertising campaigns. In video advertising campaigns the main KPI is the View Through Rate (VTR), which measures the ratio of videos ads which have been fully viewed.

The goal of this project is to define a model to predict the VTR variable using the following dataset:
  - A training file with 10k entries. For each of them is provided the VTR associated to a group of variables whose description can be found below.
 
Once the model is trained with the previous dataset, the program predicts the VTR variable for the testing set:
  - The testing file includes 2,5k entries similar to those in the training file but without the VTR.
 
 ## Relevant Information
 
The files include the following variables: appType, creatSize, creatType, deviceOs, domain, tsHour, tsDow, *_video_start, *_video_complete, *_vtr. And the variable to predict: VTR.
Where * can be: appType, creatSize, creatType, deviceOs, domain, tsHour or tsDow.

Note that VTR is computed as: *_video_complete/*_video_start. (i.e., number of users that fully viewed the video ad / number of users that started the video ad).

The schema of the data is the following one:
  - appType: string. Description: Whether it is an app or website (values: app/site)
  - creatSize: string. Description: Ad size (e.g., 300x250)
  - creatType: string. Description: Ad type (e.g, banner, video, etc.)
  - deviceOs: string. Description: Operating system of the device where the ad is
  displayed.
  - domain: string. Description: URL/website where the is displayed.
  - tsHour: string. Description: Day hour when the ad takes place.
  - tsDow: string. Description: Week day when the ad takes place.
  - appType_video_start: long
  - appType_video_complete: long
  - appType_vtr: double
  - creatSize_video_start: long
  - creatSize_video_complete: long
  - creatSize_vtr: double
  - creatType_video_start: long
  - creatType_video_complete: long
  - creatType_vtr: double
  - deviceOs_video_start: long
  - deviceOs_video_complete: long
  - deviceOs_vtr: double
  - domain_video_start: long
  - domain_video_complete: long
  - domain_vtr: double
  - tsHour_video_start: long
  - tsHour_video_complete: long
  - tsHour_vtr: double
  - tsDow_video_start: long
  - tsDow_video_complete: long
  - tsDow_vtr: double
  - VTR: double. Description: View Through Rate value
