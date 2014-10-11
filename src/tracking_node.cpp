#define BOOST_SIGNALS_NO_DEPRECATION_WARNING
#include "tracking_node.hpp"
#include "handkp_leap_msg.h"
//#include "tracking/Hand_XYZRGB.h"
#include "articulate_HandModel_XYZRGB.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <omp.h>
#include "para_deque.h"

#define PI 3.14159265

using namespace image_transport;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace cv;
using namespace image_geometry;
using namespace pcl;
using namespace Eigen;
using namespace std;
//using namespace tf2;

float half_palm_length = 0.07;
float finger_length = 0.07;
int resolution = 3;
//30cm *30cm for hand
int imageSize = 300/resolution;
bool flag_leap = true;
//pcl::PointXYZRGB my_hand_kp[31];
//pcl::PointXYZRGB last_joints_position[26];
float para_lastFrame[27], det_para[27];
int seq = 0;
int back_groud_value = 100;
para_deque parameter_que(7);

articulate_HandModel_XYZRGB MyHand;

tracking_Node::tracking_Node(ros::NodeHandle& nh):
    imageTransport_(nh),
    timeSynchronizer_(10)
  //reconfigureServer_(ros::NodeHandle(nh,"tracking")),
  //transformListener_(buffer_, true)
  //reconfigureCallback_(boost::bind(&tracking_Node::updateConfig, this, _1, _2))
{

    hand_kp_Subscriber_.subscribe(nh, "/Hand_kp_cl", 10);
    hand_Subscriber_.subscribe(nh, "/Hand_pcl", 10);
    registered_Depth_Image_Subscriber.subscribe(nh, "/Depth_Image", 10);

    timeSynchronizer_.connectInput(hand_kp_Subscriber_, hand_Subscriber_, registered_Depth_Image_Subscriber);

    segmented_hand_ = nh.advertise<sensor_msgs::PointCloud2>("Segmented_Hand",0);

    handPublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Hand_cloud",0);
    articulatePublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Articulate",0);
    modelPublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Model_point_cloud",0);

    bone_pub_ = nh.advertise<visualization_msgs::Marker>("Bones", 0);
    //leap_articulate_pub_ = nh.advertise<visualization_msgs::Marker>("Leap_Articulate",0);

    timeSynchronizer_.registerCallback(boost::bind(&tracking_Node::syncedCallback, this, _1, _2, _3));

    srand( (unsigned)time( NULL ) );
    //reconfigureServer_.setCallback(reconfigureCallback_);


}

//void  SlamNode::updateConfig(pixel_slam::slamConfig &config, uint32_t level){
//    slam_.reset(new Slam(config.min_depth,config.max_depth,config.line_requirement));
//    line_requirement_=config.line_requirement;
//    if(stereoCameraModel_.initialized()){
//        min_disparity_=stereoCameraModel_.getDisparity(config.max_depth);
//        max_disparity_=stereoCameraModel_.getDisparity(config.min_depth);
//    }
//}

void tracking_Node::syncedCallback(const PointCloud2ConstPtr& hand_kp_pter, const PointCloud2ConstPtr& hand_pter, const ImageConstPtr& cvpointer_depthImage){

    ros::Time time0 = ros::Time::now();
    cv_bridge::CvImagePtr cvpointer_depthFrame;

    int tracking_mode = 13;



    pcl::PointCloud<pcl::PointXYZRGB> msg_pcl, handcloud, hand_kp;

    ROS_INFO("Callback begins");

    try
    {
        //********************************************************//
        //1.1 get ready of Hand point cloud and joints position
        // hand point cloud in pcl::PointCloud<pcl::PointXYZRGB> handcloud; and joints position in: hand_kp;
        ros::Time time1 = ros::Time::now();

        fromROSMsg(*hand_kp_pter, hand_kp);
        fromROSMsg(*hand_pter, msg_pcl);

        //std::cout<<"Hand Keypoint size: "<<hand1_kp.size()<<endl;
        //std::cout<<"Hand Cloud size: "<<msg_pcl.size()<<endl;

        cvpointer_depthFrame = cv_bridge::toCvCopy(cvpointer_depthImage);
        Mat Origional_depthImage;
        Origional_depthImage = cvpointer_depthFrame->image;

        ros::Time time2 = ros::Time::now();

        for (size_t i = 0; i < msg_pcl.points.size (); ++i){
            if( (abs(msg_pcl.points[i].x) < 0.2
                 && abs(msg_pcl.points[i].y) < 0.2
                 && abs(msg_pcl.points[i].z) < 0.2)
                    &&((abs(msg_pcl.points[i].x)*abs(msg_pcl.points[i].x)+ abs(msg_pcl.points[i].y)*abs(msg_pcl.points[i].y)+abs(msg_pcl.points[i].z)*abs(msg_pcl.points[i].z)< half_palm_length*half_palm_length) ||
                       (abs(msg_pcl.points[i].x - hand_kp.points[1].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[1].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[1].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[7].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[7].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[7].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[13].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[13].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[13].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[19].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[19].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[19].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand_kp.points[25].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand_kp.points[25].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand_kp.points[25].z) < finger_length))
                    ){
                //                uint32_t rgb = *reinterpret_cast<int*>(&msg_pcl.points[i].rgb);
                //                uint8_t r = (rgb >> 16) & 0x0000ff;
                //                uint8_t g = (rgb >> 8) & 0x0000ff;
                //                uint8_t b = (rgb) & 0x0000ff;
                //                if(r > b && r > 70){

                handcloud.push_back(msg_pcl.points[i]);
            }
        }

        ros::Time time3 = ros::Time::now();
        //******** 1.1 done  **************//

        //********************************************************//
        //1.2 Do segmentation
        // Segmentation result in pcl::PointCloud<pcl::PointXYZRGB> Segmented_hand_cloud
        pcl::PointCloud<pcl::PointXYZRGB> Segmented_hand_cloud;
        vector<int> label;
        //GridGraphSeqeratePalm(hand_kp, handcloud, Segmented_hand_cloud, label);
        NearestNeighbour(hand_kp, handcloud, Segmented_hand_cloud, label);
        //******** 1.2 done  **************//
        ros::Time time4 = ros::Time::now();


        //******** 1.3 Ray tracing for Oberservation **********//
        // Determin whether point in visible or not, and generate visibility map stored in visibilityMap_Oberservation and visiblityMap_Hypo;
        Mat visibilityMap_Oberservation(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
        Ray_tracing_OrthognalProjection(handcloud, imageSize, resolution, visibilityMap_Oberservation);
        //******** 1.3 done *****************//
        ros::Time time5 = ros::Time::now();
        std::cout << time1-time0 << " seconds: " << "from callback begin to try." << std::endl;
        std::cout << time2-time1 << " seconds: " <<"message from transfer." <<std::endl;
        std::cout << time3-time2 << " seconds: " << "Picking hand points." << std::endl;
        std::cout << time4-time3 << " seconds: NN Segmentation." << std::endl;
        std::cout << time5-time4 << " seconds: Ray tracking for Observation." << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB> modelPointCloud;

        //******** 2.0 Point cloud hand model visulization *********//
        if(tracking_mode == -1){
            float para[27];
            float para_suboptimal[27];
            float temp_parameters[27]= {0,0,0,
                                        0,0,0,
                                        70,-40,60,20,
                                        30,0,10,10,
                                        0,10,20,10,
                                        -10,60,60,20,
                                        -20,70,70,10,
                                       70};

            for(int i = 0; i<27;i++){
                para_lastFrame[i] = temp_parameters[i];
                det_para[i] = 0;
                para[i] = temp_parameters[i];
                para_suboptimal[i] = temp_parameters[i];
            }
            MyHand.set_parameters(para);
            MyHand.get_joints_positions();
            ros::Time time_begin = ros::Time::now();
            //MyHand.get_handPointCloud(modelPointCloud);
            ros::Time time_end = ros::Time::now();
            //std::cout << time_end-time_begin << " seconds: Transform point cloud." <<std::endl;
            time_begin = ros::Time::now();
            MyHand.samplePointCloud(modelPointCloud);
            time_end = ros::Time::now();
            std::cout << time_end-time_begin << " seconds: Resample point cloud." <<std::endl;

            //******** 2.1 Projection(visibility map) ******//
            time_begin = ros::Time::now();
            Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
            Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
            time_end = ros::Time::now();
            std::cout << time_end-time_begin << " seconds: Projection of  point cloud." <<std::endl;
            //******** 2.1 done ******//

            //******** 2.2 Score (similarity assessment) ******//
            time_begin = ros::Time::now();
            float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
            Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
            time_end = ros::Time::now();
            std::cout << time_end-time_begin << " seconds: Scoring." <<std::endl;
            //******** 2.2 done *************//
        }

        //******** 2.0 particle filters for tracking --------Naive particle filter*********//
        if(tracking_mode == 0){
            float temp_para[27];
            float temp_score = 100000;

#pragma omp for
            for (int iterator = 0; iterator < 350; iterator ++){
                //******** 2.1 generate Hypothesis point cloud *******//
                float para[27];
                if(!seq){
                    float temp_parameters[27]= {0,0,0,
                                                -30,0,-10,
                                                30,10,10,0,
                                                10,0,0,0,
                                                0,0,0,0,
                                                -10,0,0,0,
                                                -20,0,0,0,};

                    for(int i = 0; i<27;i++){
                        para_lastFrame[i] = temp_parameters[i];
                        det_para[i] = 0;
                        para[i] = temp_parameters[i];
                    }
                }
                else{
                    for(int i = 0; i<27;i++){
                        para[i] = para_lastFrame[i] + det_para[i];
                    }
                }
                if (iterator == 0){
                    for(int i = 0; i< 27; i++){
                        std::cout << "para" << i <<": " << temp_para[i] << std::endl;
                    }
                }
                int step_large = 20;
                int step_small = 15;
                int a_large = step_large*100;
                int a_small = step_small*100;
                float b_large = step_large/2.0;
                float b_small = step_small/2.0;
                para[3] += rand()%a_large/100.0-b_large;
                para[4] += rand()%a_large/100.0-b_large;
                para[5] += rand()%a_large/100.0-b_large;
                para[6] += rand()%a_small/100.0-b_small;
                para[7] += rand()%a_small/100.0-b_small;
                para[8] += rand()%a_small/100.0-b_small;
                para[9] += rand()%a_small/100.0-b_small;
                para[10] += rand()%a_small/100.0-b_small;
                para[11] += rand()%a_small/100.0-b_small;
                para[12] += rand()%a_small/100.0-b_small;
                para[13] += rand()%a_small/100.0-b_small;
                para[14] += rand()%a_small/100.0-b_small;
                para[15] += rand()%a_small/100.0-b_small;
                para[16] += rand()%a_small/100.0-b_small;
                para[17] += rand()%a_small/100.0-b_small;
                para[18] += rand()%a_small/100.0-b_small;
                para[19] += rand()%a_small/100.0-b_small;
                para[20] += rand()%a_small/100.0-b_small;
                para[21] += rand()%a_small/100.0-b_small;
                para[22] += rand()%a_small/100.0-b_small;
                para[23] += rand()%a_small/100.0-b_small;
                para[24] += rand()%a_small/100.0-b_small;
                para[25] += rand()%a_small/100.0-b_small;

                MyHand.set_parameters(para);
                MyHand.get_joints_positions();
                MyHand.get_handPointCloud(modelPointCloud);
                //******** 2.1 done ****************//

                //******** 2.2 Ray tracing for Hypothesis ********//
                Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                //******** 2.2 done *******************//

                //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                //        waitKey();

                ROS_INFO("Prepare Model Cloud");
                sensor_msgs::PointCloud2 model_cloud_msg;
                toROSMsg(visibleModelPointCloud,model_cloud_msg);
                model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                modelPublisher_.publish(model_cloud_msg);

                //******** 2.3 Score (similarity assessment) ******//
                float score;
                Score(visibilityMap_Oberservation, visiblityMap_Hypo, score);
                //******** 2.3 done *************//

                std::cout << "Score: " << score << std::endl;

                if(score < temp_score){
                    temp_score = score;
                    for(int i = 0; i< 27; i++)
                        temp_para[i] = para[i];


                }
            }
            for(int i = 0; i< 27; i++){
                std::cout << "para" << i <<": " << temp_para[i] << std::endl;
                det_para[i] = temp_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = temp_para[i];
            }
            std::cout << "Error: " <<temp_score << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(temp_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------annealing as a whole *********//
        else if (tracking_mode == 1){
            float optimal_para[27];
            float optimal_score = 100000;
            float Opt_Score_aver_overlapdiff = 10000;
            for( int annealling_iterator = 0; annealling_iterator < 5; annealling_iterator++){
                float annealling_factor = pow(2,-annealling_iterator);
                float para[27];
                float para_suboptimal[27];
                if(!seq){
                    if(!annealling_iterator){
                        float temp_parameters[27]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,
                                                   70};

                        for(int i = 0; i<27;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    else{
                        for(int i = 0; i<27; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = optimal_para[i];
                        }
                    }
                }
                else{
                    if( !annealling_iterator){
                        for(int i = 0; i<27;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    else{
                        for(int i = 0; i<27; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                }

#pragma omp for
                for (int iterator = 0; iterator < 80; iterator ++){
                    //******** 2.1 generate Hypothesis point cloud *******//

                    if (iterator == 0){
                        for(int i = 0; i< 27; i++){
                            std::cout << "para" << i <<": " << para[i] << std::endl;
                        }
                    }
                    int step_large = 20;
                    int step_small = 10;
                    int a_large = step_large*100;
                    int a_small = step_small*100;
                    float b_large = step_large/2.0;
                    float b_small = step_small/2.0;


                    para[3] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[4] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[5] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[6] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[7] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[8] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[9] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[10] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[11] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[12] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[13] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[14] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[15] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[16] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[17] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[18] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[19] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[20] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[21] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[22] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[23] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[24] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[25] += (rand()%a_small/100.0-b_small)*annealling_factor;

                    for(int i = 0; i < 27; i++){
                        if(para[i] > MyHand.parameters_max[i]){
                            para[i] = MyHand.parameters_max[i];
                        }
                        else if (para[i] < MyHand.parameters_min[i]){
                            para[i] = MyHand.parameters_min[i];
                        }
                    }

                    MyHand.set_parameters(para);
                    MyHand.get_joints_positions();
                    MyHand.get_handPointCloud(modelPointCloud);
                    //******** 2.1 done ****************//

                    //******** 2.2 Ray tracing for Hypothesis ********//
                    Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                    pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                    Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                    //******** 2.2 done *******************//

                    //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                    //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                    //        waitKey();



                    //                    //******** 2.3 Score (similarity assessment) ******//
                    //                    float score;
                    //                    Score(visibilityMap_Oberservation, visiblityMap_Hypo, score);
                    //                    float overlap, overall_diff, overlap_diff;
                    //                    Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overall_diff, overlap_diff);
                    //                    std::cout << "Multi score: " << overlap << "  " << overall_diff << "  " << overlap_diff << std::endl;
                    //                    //******** 2.3 done *************//

                    //                    std::cout << "Score: " << score << std::endl;

                    //                    if(score < optimal_score){
                    //                        optimal_score = score;
                    //                        for(int i = 0; i< 27; i++){
                    //                            optimal_para[i] = para[i];
                    //                        }
                    //                    }
                    //******** 2.3 Score (similarity assessment) ******//
                    float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                    Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                    //******** 2.3 done *************//

                    //std::cout << "Overall_diff: " << overall_diff << std::endl;

                    if((overlap_obs + overlap_hyp)*1.0/overlap <= optimal_score && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                        optimal_score = (overlap_obs + overlap_hyp)*1.0/overlap;
                        Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                        for(int i = 0; i< 27; i++){
                            optimal_para[i] = para[i];
                        }
                    }
                    else{

                    }

                    for(int i = 0; i< 27; i++){
                        //std::cout << "para" << i <<": " << para[i] << std::endl;
                        para[i] = para_suboptimal[i];
                    }

                }
                for(int i = 0; i< 27; i++){
                    std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                    para_suboptimal[i] = optimal_para[i];
                }
                MyHand.set_parameters(optimal_para);
                MyHand.get_joints_positions();
                MyHand.get_handPointCloud(modelPointCloud);

                ROS_INFO("Prepare Model Cloud");
                sensor_msgs::PointCloud2 model_cloud_msg;
                toROSMsg(modelPointCloud,model_cloud_msg);
                model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                modelPublisher_.publish(model_cloud_msg);

                std::cout << "Error: " <<optimal_score << std::endl;
            }

            for(int i = 0; i< 27; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                //det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
            }

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------kinematic chain tracking *********//
        else if (tracking_mode == 2){

            float optimal_para[26];
            float temp_score = 100000;

            for(int kinematic_chain = 0; kinematic_chain < 3; ++kinematic_chain){
                float para[26];
                float para_suboptimal[26];
                if(!seq){
                    if(!kinematic_chain){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                }
                else{
                    if( !kinematic_chain){
                        for(int i = 0; i<26;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                }

#pragma omp for
                for (int iterator = 0; iterator < 80; iterator ++){
                    //******** 2.1 generate Hypothesis point cloud *******//

                    if (iterator == 0){
                        for(int i = 0; i< 26; i++){
                            std::cout << "para" << i <<": " << para[i] << std::endl;
                        }
                    }
                    else{

                    }
                    int step_large = 10;
                    int step_small = 10;
                    int a_large = step_large*100;
                    int a_small = step_small*100;
                    float b_large = step_large/2.0;
                    float b_small = step_small/2.0;

                    if(kinematic_chain == 0){
                        para[3] += rand()%a_large/100.0-b_large;
                        para[4] += rand()%a_large/100.0-b_large;
                        para[5] += rand()%a_large/100.0-b_large;
                    }
                    else if (kinematic_chain == 1){
                        para[6] += rand()%a_small/100.0-b_small;
                        para[7] += rand()%a_large/100.0-b_large;

                        para[10] += rand()%a_small/100.0-b_small;
                        para[11] += rand()%a_large/100.0-b_large;

                        para[14] += rand()%a_small/100.0-b_small;
                        para[15] += rand()%a_large/100.0-b_large;

                        para[18] += rand()%a_small/100.0-b_small;
                        para[19] += rand()%a_large/100.0-b_large;

                        para[22] += rand()%a_small/100.0-b_small;
                        para[23] += rand()%a_large/100.0-b_large;
                    }
                    else{
                        para[8] += rand()%a_large/100.0-b_large;
                        para[9] += rand()%a_small/100.0-b_small;

                        para[12] += rand()%a_large/100.0-b_large;
                        para[13] += rand()%a_small/100.0-b_small;

                        para[16] += rand()%a_large/100.0-b_large;
                        para[17] += rand()%a_small/100.0-b_small;

                        para[20] += rand()%a_large/100.0-b_large;
                        para[21] += rand()%a_small/100.0-b_small;

                        para[24] += rand()%a_large/100.0-b_large;
                        para[25] += rand()%a_small/100.0-b_small;
                    }
                    //check parameters:
                    for(int i = 0; i < 26; i++){
                        if(para[i] > MyHand.parameters_max[i]){
                            para[i] = MyHand.parameters_max[i];
                        }
                        else if (para[i] < MyHand.parameters_min[i]){
                            para[i] = MyHand.parameters_min[i];
                        }
                    }

                    MyHand.set_parameters(para);
                    MyHand.get_joints_positions();
                    MyHand.get_handPointCloud(modelPointCloud);
                    //******** 2.1 done ****************//

                    //******** 2.2 Ray tracing for Hypothesis ********//
                    Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                    pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                    Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                    //******** 2.2 done *******************//

                    //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                    //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                    //        waitKey();

                    ROS_INFO("Prepare Model Cloud");
                    sensor_msgs::PointCloud2 model_cloud_msg;
                    toROSMsg(visibleModelPointCloud,model_cloud_msg);
                    model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                    model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                    modelPublisher_.publish(model_cloud_msg);

                    //******** 2.3 Score (similarity assessment) ******//
                    float score;
                    Score(visibilityMap_Oberservation, visiblityMap_Hypo, score);
                    //******** 2.3 done *************//

                    std::cout << "Score: " << score << std::endl;

                    if(score < temp_score){
                        temp_score = score;
                        for(int i = 0; i< 26; i++){
                            optimal_para[i] = para[i];
                        }
                    }

                    for(int i = 0; i< 26; i++){
                        std::cout << "para" << i <<": " << para[i] << std::endl;
                        para[i] = para_suboptimal[i];
                    }
                }
            }
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
            }
            std::cout << "Error: " <<temp_score << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------annealing kinematic chain with multi score *****//
        else if (tracking_mode == 3){
            float optimal_para[26];
            float Opt_Score_overalldiff = 100000, Opt_Score_overlapdiff = 100000, Opt_Score_overlap = 0;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 3; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

#pragma omp for
                    for (int iterator = 0; iterator < 50; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        if (iterator == 0){
                            for(int i = 0; i< 26; i++){
                                std::cout << "para" << i <<": " << para[i] << std::endl;
                            }
                        }
                        else{

                        }
                        int step_large = 15;
                        int step_small = 10;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            para[3] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[4] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                            para[7] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                            para[11] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                            para[15] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                            para[19] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                            para[23] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else{
                            para[8] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[9] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[12] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[13] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[16] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[17] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[20] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[21] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[24] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[25] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        //check parameters:
                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        MyHand.get_handPointCloud(modelPointCloud);
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        ROS_INFO("Prepare Model Cloud");
                        sensor_msgs::PointCloud2 model_cloud_msg;
                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        std::cout << "Overall_diff: " << overall_diff << std::endl;
                        if(annealing_iterator < 10){
                            if(overall_diff < Opt_Score_overalldiff && overlap_diff < Opt_Score_overlapdiff){
                                Opt_Score_overalldiff = overall_diff;
                                Opt_Score_overlapdiff = overlap_diff;
                                for(int i = 0; i< 26; i++){
                                    optimal_para[i] = para[i];
                                }
                            }
                        }
                        else{
                            if(overlap_diff < Opt_Score_overlapdiff){
                                Opt_Score_overlapdiff = overlap_diff;
                                for(int i = 0; i< 26; i++){
                                    optimal_para[i] = para[i];
                                }
                            }
                        }

                        for(int i = 0; i< 26; i++){
                            std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
            }
            std::cout << "Error: " <<Opt_Score_overalldiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------annealing one by one kinematic chain with multi score *****//
        else if (tracking_mode == 4){
            float optimal_para[26];
            float Opt_Score_overalldiff = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlap = 0;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 40;
                    }

#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        if (iterator == 0){
                            for(int i = 0; i< 26; i++){
                                std::cout << "para" << i <<": " << para[i] << std::endl;
                            }
                        }
                        else{

                        }
                        int step_large = 15;
                        int step_small = 10;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            para[3] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[4] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }

                        else{
                            para[kinematic_chain+5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        //check parameters:
                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        MyHand.get_handPointCloud(modelPointCloud);
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        ROS_INFO("Prepare Model Cloud");
                        sensor_msgs::PointCloud2 model_cloud_msg;
                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if(overall_diff < Opt_Score_overalldiff && overlap_diff/overlap < Opt_Score_aver_overlapdiff){
                            Opt_Score_overalldiff = overall_diff;
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 26; i++){
                            std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
            }
            std::cout << "Error: " <<Opt_Score_overalldiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------annealing one by one kinematic chain with multi score (overlap rate)*****//
        else if (tracking_mode == 5){
            float optimal_para[26];
            float Opt_Score_overalldiff = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlap = 0;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 30;
                    }

#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 26; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        int step_large = 15;
                        int step_small = 10;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            para[3] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[4] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }

                        else{
                            para[kinematic_chain+5] += (rand()%a_large/100.0-b_large)*2*annealing_factor;
                        }
                        //check parameters:
                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        MyHand.get_handPointCloud(modelPointCloud);
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        sensor_msgs::PointCloud2 model_cloud_msg;
                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs + overlap_hyp)*1.0/overlap <= Opt_Score_overalldiff && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                            Opt_Score_overalldiff = (overlap_obs + overlap_hyp)*1.0/overlap;
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 26; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
            }
            std::cout << "Error: " <<Opt_Score_overalldiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------annealing one by one kinematic chain with multi-score (overlap rate) and parameter adjustment*****//
        else if (tracking_mode == 6){
            float optimal_para[26];
            float Opt_Score_overalldiff = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlap = 0;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 30;
                    }

#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 26; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        int step_large = 15;
                        int step_small = 10;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            para[3] += ((para[11]+para[15]+para[19]+para[23])/8.0);
                            para[3] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[4] += (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }

                        else{
                            para[kinematic_chain+5] += (rand()%a_large/100.0-b_large)*2*annealing_factor;
                        }
                        //check parameters:
                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        MyHand.get_handPointCloud(modelPointCloud);
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        sensor_msgs::PointCloud2 model_cloud_msg;
                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs + overlap_hyp)*1.0/overlap <= Opt_Score_overalldiff && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                            Opt_Score_overalldiff = (overlap_obs + overlap_hyp)*1.0/overlap;
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 26; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
            }
            std::cout << "Overlap ratio: " <<Opt_Score_overalldiff << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------annealing one by one kinematic chain with multi-score (overlap rate) and local parameter influence*****//
        else if (tracking_mode == 7){
            float optimal_para[27];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[27];
                    float para_suboptimal[27];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[27]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,
                                                   70};

                        for(int i = 0; i<27;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<27;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<27; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 30;
                    }

#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 27; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        int step_large = 15;
                        int step_small = 10;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            float temp;
                            temp = 2*(rand()%a_large/100.0-b_large)*annealing_factor;
                            para[3] += temp;
                            para[7] -= 0.8*temp;
                            para[11] -= 0.8*temp;
                            para[15] -= 0.8*temp;
                            para[19] -= 0.8*temp;
                            para[23] -= 0.8*temp;

                            para[4] += (rand()%a_large/100.0-b_large)*2*annealing_factor;

                            para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }

                        else{
                            float temp = (rand()%a_large/100.0-b_large)*3*annealing_factor;
                            para[kinematic_chain+5] += temp;
                            if((kinematic_chain+6-2)%4 != 0)
                                para[kinematic_chain+6] -= 0.8*temp;
                            if(kinematic_chain+5%4 == 1){
                                if(para[kinematic_chain+6] > para[kinematic_chain+5])
                                    para[kinematic_chain+5] += 0.8*(para[kinematic_chain+6] - para[kinematic_chain+5]);
                            }
                            //                            if(para[9] > para[8])
                            //                                para[8] += 0.8*(para[9]-para[8]);
                            //                            if(para[13] > para[12])
                            //                                para[12] += 0.8*(para[13]-para[12]);
                            //                            if(para[17] > para[16])
                            //                                para[16] += 0.8*(para[17]-para[16]);
                            //                            if(para[21] > para[20])
                            //                                para[20] += 0.8*(para[21]-para[20]);
                            //                            if(para[25] > para[24])
                            //                                para[24] += 0.8*(para[25]-para[24]);
                        }
                        //check parameters:

                        for(int i = 0; i < 27; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        MyHand.get_handPointCloud(modelPointCloud);
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        sensor_msgs::PointCloud2 model_cloud_msg;
                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 27; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 27; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            vector<float> for_the_que;
            for(int i = 0; i< 27; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
                for_the_que.push_back(optimal_para[i]);
            }
            parameter_que.add_new(for_the_que);
            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------7 with probablistic acceptence*****//
        else if (tracking_mode == 8){
            float optimal_para[26];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            para[i] = para_lastFrame[i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 30;
                    }

#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 26; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        int step_large = 15;
                        int step_small = 10;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            float temp;
                            temp = (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[3] += temp;
                            para[11] -= temp;
                            para[15] -= temp;
                            para[19] -= temp;
                            para[23] -= temp;

                            para[4] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }

                        else{
                            float temp = (rand()%a_large/100.0-b_large)*3*annealing_factor;
                            para[kinematic_chain+5] += temp;
                            if((kinematic_chain+6-2)%4 != 0)
                                para[kinematic_chain+6] -= 0.8*temp;
                            if(kinematic_chain+5%4 == 1){
                                if(para[kinematic_chain+6] > para[kinematic_chain+5])
                                    para[kinematic_chain+5] += 0.8*(para[kinematic_chain+6] - para[kinematic_chain+5]);
                            }
                            //                            if(para[9] > para[8])
                            //                                para[8] += 0.8*(para[9]-para[8]);
                            //                            if(para[13] > para[12])
                            //                                para[12] += 0.8*(para[13]-para[12]);
                            //                            if(para[17] > para[16])
                            //                                para[16] += 0.8*(para[17]-para[16]);
                            //                            if(para[21] > para[20])
                            //                                para[20] += 0.8*(para[21]-para[20]);
                            //                            if(para[25] > para[24])
                            //                                para[24] += 0.8*(para[25]-para[24]);
                        }
                        //check parameters:

                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        MyHand.get_handPointCloud(modelPointCloud);
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        sensor_msgs::PointCloud2 model_cloud_msg;
                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        int condition_score = 0;
                        if(overlap_obs/overlap <= Opt_Score_overlapratio1)
                            condition_score += 1;
                        if(overlap_hyp/overlap <= Opt_Score_overlapratio2)
                            condition_score += 1;
                        if(overlap_diff/overlap <= Opt_Score_aver_overlapdiff)
                            condition_score += 1;

                        std::cout << "Condition Score: " << condition_score << std::endl;

                        bool accept_flag = false;
                        if(condition_score == 3)
                            accept_flag = true;
                        else if (condition_score ==2){
                            if(rand()%10 < 9)
                                accept_flag = true;
                        }
                        else if (condition_score == 1){
                            if(rand()%100 < 1)
                                accept_flag = true;
                        }

                        if(accept_flag == true){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                            std::cout<< "Accepted" << std::endl;
                        }


                        for(int i = 0; i< 26; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                det_para[i] = optimal_para[i]-para_lastFrame[i];
                det_para[i] = 0;
                para_lastFrame[i] = optimal_para[i];
            }
            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------7 with sample model generation*****//
        else if (tracking_mode == 9){
            float optimal_para[26];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            para[i] = parameter_que.para_sequence_smoothed[0][i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 30;
                    }
                    //#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 26; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        int step_large = 10;
                        int step_small = 10;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            float temp;
                            temp = (rand()%a_large/100.0-b_large)*annealing_factor;
                            para[3] += temp;
                            para[7] -= 0.8*temp;
                            para[11] -= 0.8*temp;
                            para[15] -= 0.8*temp;
                            para[19] -= 0.8*temp;
                            para[23] -= 0.8*temp;

                            para[4] += (rand()%a_large/100.0-b_large)*annealing_factor;

                            para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }

                        else{
                            float temp = (rand()%a_large/100.0-b_large)*3*annealing_factor;
                            para[kinematic_chain+5] += temp;
                            if((kinematic_chain+6-2)%4 != 0)
                                para[kinematic_chain+6] -= 0.8*temp;
                            if(kinematic_chain+5%4 == 1){
                                if(para[kinematic_chain+6] > para[kinematic_chain+5])
                                    para[kinematic_chain+5] += 0.8*(para[kinematic_chain+6] - para[kinematic_chain+5]);
                            }
                            //                            if(para[9] > para[8])
                            //                                para[8] += 0.8*(para[9]-para[8]);
                            //                            if(para[13] > para[12])
                            //                                para[12] += 0.8*(para[13]-para[12]);
                            //                            if(para[17] > para[16])
                            //                                para[16] += 0.8*(para[17]-para[16]);
                            //                            if(para[21] > para[20])
                            //                                para[20] += 0.8*(para[21]-para[20]);
                            //                            if(para[25] > para[24])
                            //                                para[24] += 0.8*(para[25]-para[24]);
                        }
                        //check parameters:

                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        //ros::Time time_begin = ros::Time::now();
                        //MyHand.get_handPointCloud(modelPointCloud);
                        MyHand.samplePointCloud(modelPointCloud);
                        //ros::Time time_end = ros::Time::now();
                        //std::cout<<"Duration: "<< time_end-time_begin << std::endl;
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        //Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        //                        sensor_msgs::PointCloud2 model_cloud_msg;
                        //                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        //                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        //                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        //                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 26; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            vector<float> for_the_que;
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                para_lastFrame[i] = optimal_para[i];
                for_the_que.push_back(optimal_para[i]);
            }
            parameter_que.add_new(for_the_que);
            parameter_que.smooth_mean(3);
            for(int i = 0; i< 26; i++){
                det_para[i] = parameter_que.para_delta[i];
            }

            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            for(int i = 0; i< 25; i++){
                optimal_para[i] = parameter_que.para_sequence_smoothed[0][i];
            }
            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            //MyHand.get_handPointCloud(modelPointCloud);
            MyHand.samplePointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------annealing as a whole and then annealing one by one kinematic chain*****//
        else if (tracking_mode == 10){
            //annealing as a whole first
            float optimal_para[26];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for( int annealling_iterator = 0; annealling_iterator < 2; annealling_iterator++){
                float annealling_factor = pow(2,-annealling_iterator);
                float para[26];
                float para_suboptimal[26];
                if(!seq){
                    if(!annealling_iterator){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = optimal_para[i];
                        }
                    }
                }
                else{
                    if( !annealling_iterator){
                        for(int i = 0; i<26;i++){
                            para[i] = parameter_que.para_sequence_smoothed[0][i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                }

#pragma omp for
                for (int iterator = 0; iterator < 80; iterator ++){
                    //******** 2.1 generate Hypothesis point cloud *******//

                    if (iterator == 0){
                        for(int i = 0; i< 26; i++){
                            std::cout << "para" << i <<": " << para[i] << std::endl;
                        }
                    }
                    int step_large = 30;
                    int step_small = 20;
                    int a_large = step_large*100;
                    int a_small = step_small*100;
                    float b_large = step_large/2.0;
                    float b_small = step_small/2.0;


                    para[3] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[4] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[5] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[6] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[7] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[8] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[9] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[10] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[11] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[12] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[13] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[14] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[15] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[16] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[17] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[18] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[19] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[20] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[21] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[22] += (rand()%a_small/100.0-b_small)*annealling_factor;
                    para[23] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[24] += (rand()%a_large/100.0-b_large)*annealling_factor;
                    para[25] += (rand()%a_large/100.0-b_large)*annealling_factor;

                    for(int i = 0; i < 26; i++){
                        if(para[i] > MyHand.parameters_max[i]){
                            para[i] = MyHand.parameters_max[i];
                        }
                        else if (para[i] < MyHand.parameters_min[i]){
                            para[i] = MyHand.parameters_min[i];
                        }
                    }

                    MyHand.set_parameters(para);
                    MyHand.get_joints_positions();
                    MyHand.get_handPointCloud(modelPointCloud);
                    //******** 2.1 done ****************//

                    //******** 2.2 Ray tracing for Hypothesis ********//
                    Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                    pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                    Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                    //******** 2.2 done *******************//


                    //******** 2.3 Score (similarity assessment) ******//
                    float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                    Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                    //******** 2.3 done *************//

                    //std::cout << "Overall_diff: " << overall_diff << std::endl;

                    if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                        Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                        Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                        Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                        for(int i = 0; i< 26; i++){
                            optimal_para[i] = para[i];
                        }
                    }

                    for(int i = 0; i< 26; i++){
                        //std::cout << "para" << i <<": " << para[i] << std::endl;
                        para[i] = para_suboptimal[i];
                    }

                }
                for(int i = 0; i< 26; i++){
                    std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                    para_suboptimal[i] = optimal_para[i];
                }
                //                    MyHand.set_parameters(optimal_para);
                //                    MyHand.get_joints_positions();
                //                    MyHand.get_handPointCloud(modelPointCloud);

                //                    ROS_INFO("Prepare Model Cloud");
                //                    sensor_msgs::PointCloud2 model_cloud_msg;
                //                    toROSMsg(modelPointCloud,model_cloud_msg);
                //                    model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                //                    model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                //                    modelPublisher_.publish(model_cloud_msg);

                std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
                std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
                std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;
            }

            //                for(int i = 0; i< 26; i++){
            //                    std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
            //                    //det_para[i] = optimal_para[i]-para_lastFrame[i];
            //                    //det_para[i] = 0;
            //                    //para_lastFrame[i] = optimal_para[i];
            //                }

            //                MyHand.set_parameters(optimal_para);
            //                MyHand.get_joints_positions();
            //                MyHand.get_handPointCloud(modelPointCloud);


            //annealing kinematic chain:
            for(int annealing_iterator = 2; annealing_iterator < 4; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];

                    //use last frame result for current frame initialization
                    if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            //para[i] = parameter_que.para_sequence_smoothed[0][i] + det_para[i];
                            //para_suboptimal[i] = para[i];
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 30;
                    }

#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 26; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        int step_large = 30;
                        int step_small = 20;
                        int a_large = step_large*100;
                        int a_small = step_small*100;
                        float b_large = step_large/2.0;
                        float b_small = step_small/2.0;

                        if(kinematic_chain == 0){
                            float temp;
                            temp = 2*(rand()%a_large/100.0-b_large)*annealing_factor;
                            para[3] += temp;
                            para[7] -= 0.8*temp;
                            para[11] -= 0.8*temp;
                            para[15] -= 0.8*temp;
                            para[19] -= 0.8*temp;
                            para[23] -= 0.8*temp;

                            para[4] += (rand()%a_large/100.0-b_large)*2*annealing_factor;

                            para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%a_large/100.0-b_large)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%a_small/100.0-b_small)*annealing_factor;
                        }

                        else{
                            float temp = (rand()%a_large/100.0-b_large)*3*annealing_factor;
                            para[kinematic_chain+5] += temp;
                            if((kinematic_chain+6-2)%4 != 0)
                                para[kinematic_chain+6] -= 0.8*temp;
                            if(kinematic_chain+5%4 == 1){
                                if(para[kinematic_chain+6] > para[kinematic_chain+5])
                                    para[kinematic_chain+5] += 0.8*(para[kinematic_chain+6] - para[kinematic_chain+5]);
                            }
                            //                            if(para[9] > para[8])
                            //                                para[8] += 0.8*(para[9]-para[8]);
                            //                            if(para[13] > para[12])
                            //                                para[12] += 0.8*(para[13]-para[12]);
                            //                            if(para[17] > para[16])
                            //                                para[16] += 0.8*(para[17]-para[16]);
                            //                            if(para[21] > para[20])
                            //                                para[20] += 0.8*(para[21]-para[20]);
                            //                            if(para[25] > para[24])
                            //                                para[24] += 0.8*(para[25]-para[24]);
                        }
                        //check parameters:

                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        MyHand.get_handPointCloud(modelPointCloud);
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        sensor_msgs::PointCloud2 model_cloud_msg;
                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 26; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            vector<float> for_the_que;
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                para_lastFrame[i] = optimal_para[i];
                for_the_que.push_back(optimal_para[i]);
            }
            parameter_que.add_new(for_the_que);
            parameter_que.smooth_mean(3);
            for(int i = 0; i< 26; i++){
                det_para[i] = parameter_que.para_delta[i];
            }

            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            for(int i = 0; i< 25; i++){
                optimal_para[i] = parameter_que.para_sequence_smoothed[0][i];
            }
            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            MyHand.get_handPointCloud(modelPointCloud);


        }

        //******** 2.0 particle filters for tracking --------9 with pseudo random*****//
        else if (tracking_mode == 11){
            float optimal_para[26];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int kinematic_chain = 0; kinematic_chain < 21; ++kinematic_chain){
                    float para[26];
                    float para_suboptimal[26];
                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!kinematic_chain)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }
                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!kinematic_chain)){
                        for(int i = 0; i<26;i++){
                            para[i] = parameter_que.para_sequence_smoothed[0][i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;
                    if (kinematic_chain == 0){
                        max_iterator = 30;
                    }
                    //#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 26; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        float angle_step = 5;
                        int angle_mode = int(2*angle_step*100);

                        if(kinematic_chain == 0){
                            float temp;
                            temp = 2*(rand()%angle_mode/100.0-angle_step)*annealing_factor;
                            para[3] += temp;
                            para[7] -= 0.8*temp;
                            para[11] -= 0.8*temp;
                            para[15] -= 0.8*temp;
                            para[19] -= 0.8*temp;
                            para[23] -= 0.8*temp;

                            para[4] += 2*(rand()%angle_mode/100.0-angle_step)*annealing_factor;

                            para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += 2*(rand()%angle_mode/100.0-angle_step)*annealing_factor;
                        }
                        else if (kinematic_chain == 1){
                            para[6] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if (kinematic_chain == 5){
                            para[10] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if (kinematic_chain == 9){
                            para[14] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if (kinematic_chain == 13){
                            para[18] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if (kinematic_chain == 17){
                            para[22] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }

                        else{
                            float temp = 4*(rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                            para[kinematic_chain+5] += temp;
                            //                            if((kinematic_chain+6-2)%4 != 0)
                            //                                para[kinematic_chain+6] -= 0.8*temp;
                            //                            if(kinematic_chain+5%4 == 1){
                            //                                if(para[kinematic_chain+6] > para[kinematic_chain+5])
                            //                                    para[kinematic_chain+5] += 0.8*(para[kinematic_chain+6] - para[kinematic_chain+5]);
                            //                            }
                            //                            if(para[9] > para[8])
                            //                                para[8] += 0.8*(para[9]-para[8]);
                            //                            if(para[13] > para[12])
                            //                                para[12] += 0.8*(para[13]-para[12]);
                            //                            if(para[17] > para[16])
                            //                                para[16] += 0.8*(para[17]-para[16]);
                            //                            if(para[21] > para[20])
                            //                                para[20] += 0.8*(para[21]-para[20]);
                            //                            if(para[25] > para[24])
                            //                                para[24] += 0.8*(para[25]-para[24]);
                        }
                        //check parameters:

                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        //ros::Time time_begin = ros::Time::now();
                        //MyHand.get_handPointCloud(modelPointCloud);
                        MyHand.samplePointCloud(modelPointCloud);
                        //ros::Time time_end = ros::Time::now();
                        //std::cout<<"Duration: "<< time_end-time_begin << std::endl;
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        //Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        //                        sensor_msgs::PointCloud2 model_cloud_msg;
                        //                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        //                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        //                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        //                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 26; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            vector<float> for_the_que;
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                para_lastFrame[i] = optimal_para[i];
                for_the_que.push_back(optimal_para[i]);
            }
            parameter_que.add_new(for_the_que);
            parameter_que.smooth_mean(3);
            for(int i = 0; i< 26; i++){
                det_para[i] = parameter_que.para_delta[i];
            }

            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            for(int i = 0; i< 25; i++){
                optimal_para[i] = parameter_que.para_sequence_smoothed[0][i];
            }
            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            //MyHand.get_handPointCloud(modelPointCloud);
            MyHand.samplePointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------9 with all parameter dimension*****//
        else if (tracking_mode == 12){
            float optimal_para[26];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.5, annealing_iterator);
                for(int parameterDimension = 0; parameterDimension < 26; ++parameterDimension){
                    float para[26];
                    float para_suboptimal[26];

                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!parameterDimension)){
                        float temp_parameters[26]= {0,0,0,
                                                    -30,0,-10,
                                                    30,10,10,0,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0};

                        for(int i = 0; i<26;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }

                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!parameterDimension)){
                        for(int i = 0; i<26;i++){
                            para[i] = parameter_que.para_sequence_smoothed[0][i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<26; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 5;

                    //#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        //                        if (iterator == 0){
                        //                            for(int i = 0; i< 26; i++){
                        //                                std::cout << "para" << i <<": " << para[i] << std::endl;
                        //                            }
                        //                        }

                        float translation_step = 0.01;
                        int translation_mode = int(2*translation_step*1000);

                        int angle_step = 10;
                        int angle_mode = 2*angle_step*100;

                        if(parameterDimension < 3){
                            para[parameterDimension] += (rand()%translation_mode/1000.0-translation_step)*annealing_factor;
                        }
                        else if (parameterDimension < 5){
                            para[parameterDimension] += (rand()%angle_mode/100.0-angle_step)*annealing_factor;
                        }
                        else if (parameterDimension == 5){
                            para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%angle_mode/100.0-angle_step)*annealing_factor;
                        }
                        else if (parameterDimension == 6){
                            para[6] += (rand()%angle_mode/100.0-angle_step)*annealing_factor;
                        }
                        else {
                            para[parameterDimension] += (rand()%angle_mode/100.0-angle_step)*annealing_factor;
                        }


                        //check parameters:

                        for(int i = 0; i < 26; i++){
                            if(para[i] > MyHand.parameters_max[i]){
                                para[i] = MyHand.parameters_max[i];
                            }
                            else if (para[i] < MyHand.parameters_min[i]){
                                para[i] = MyHand.parameters_min[i];
                            }
                        }

                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        //ros::Time time_begin = ros::Time::now();
                        //MyHand.get_handPointCloud(modelPointCloud);
                        MyHand.samplePointCloud(modelPointCloud);
                        //ros::Time time_end = ros::Time::now();
                        //std::cout<<"Duration: "<< time_end-time_begin << std::endl;
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        //Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        //                        sensor_msgs::PointCloud2 model_cloud_msg;
                        //                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        //                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        //                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        //                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && overlap_diff/overlap <= Opt_Score_aver_overlapdiff){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 26; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 26; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            vector<float> for_the_que;
            for(int i = 0; i< 26; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                para_lastFrame[i] = optimal_para[i];
                for_the_que.push_back(optimal_para[i]);
            }
            parameter_que.add_new(for_the_que);
            parameter_que.smooth_mean(3);
            for(int i = 0; i< 26; i++){
                det_para[i] = parameter_que.para_delta[i];
            }

            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            for(int i = 0; i< 25; i++){
                optimal_para[i] = parameter_que.para_sequence_smoothed[0][i];
            }
            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            //MyHand.get_handPointCloud(modelPointCloud);
            MyHand.samplePointCloud(modelPointCloud);
        }

        //******** 2.0 particle filters for tracking --------12 with pseudo random*****//
        else if (tracking_mode == 13){
            float optimal_para[27];
            float Opt_Score_overlapratio1 = 100000, Opt_Score_aver_overlapdiff = 100000, Opt_Score_overlapratio2 = 10000;
            for(int annealing_iterator = 0; annealing_iterator < 3; annealing_iterator++){
                float annealing_factor = pow(0.6, annealing_iterator);
                for(int parameterDimension = 0; parameterDimension < 27; ++parameterDimension){
                    float para[27];
                    float para_suboptimal[27];

                    //very first (initialization of the whole programme)
                    if((!seq)&&(!annealing_iterator)&&(!parameterDimension)){
                        float temp_parameters[27]= {0,0,0,
                                                    -30,0,-10,
                                                    10,-30,0,10,
                                                    10,0,0,0,
                                                    0,0,0,0,
                                                    -10,0,0,0,
                                                    -20,0,0,0,
                                                   70};

                        for(int i = 0; i<27;i++){
                            para_lastFrame[i] = temp_parameters[i];
                            det_para[i] = 0;
                            para[i] = temp_parameters[i];
                            para_suboptimal[i] = temp_parameters[i];
                        }

                    }
                    //use last frame result for current frame initialization
                    else if ((!annealing_iterator)&&(!parameterDimension)){
                        for(int i = 0; i<27;i++){
                            para[i] = parameter_que.para_sequence_smoothed[0][i] + det_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }

                    //use last kinematic_chain result for current kinematic chain initialization
                    else{
                        for(int i = 0; i<27; ++i){
                            para[i] = optimal_para[i];
                            para_suboptimal[i] = para[i];
                        }
                    }
                    int max_iterator = 4;

                    float translation_step = 0.01;
                    int translation_mode = int(2*translation_step*1000);

                    float angle_step = 10;
                    int angle_mode = int(2*angle_step*100);

                    //#pragma omp for
                    for (int iterator = 0; iterator < max_iterator; iterator ++){
                        //******** 2.1 generate Hypothesis point cloud *******//

                        if(parameterDimension < 3){
                            para[parameterDimension] += (rand()%translation_mode/1000.0/max_iterator+2*translation_step/max_iterator*iterator-translation_step)*annealing_factor;
                        }
                        else if (parameterDimension < 5){
                            para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                            if(parameterDimension == 4){

                            }
                        }
                        else if (parameterDimension == 5){
                            //para[5] += ((para[10]-10)+(para[14]-0)+(para[18]+6.7))/3.0;
                            para[5] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if (parameterDimension == 6){
                            para[6] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*2;
                        }
                        else if (parameterDimension == 7){
                            para[7] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*2;
                        }
                        else if (parameterDimension == 8){
                            para[8] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor;
                        }
                        else if ( parameterDimension == 26){
                            para[26] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                        }
                        else if ( (parameterDimension -6)%4 == 0){
                            para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor/2.0;
                        }
                        else{
                            para[parameterDimension] += (rand()%angle_mode/100.0/max_iterator+2*angle_step/max_iterator*iterator-angle_step)*annealing_factor*3.0;
                        }

                        //check parameters:
                        {
                            //min and max
                            for(int i = 0; i < 27; i++){
                                if(para[i] > MyHand.parameters_max[i]){
                                    para[i] = MyHand.parameters_max[i];
                                }
                                else if (para[i] < MyHand.parameters_min[i]){
                                    para[i] = MyHand.parameters_min[i];
                                }
                            }
                            //collision between 2~5 finger
                            for(int i = 10; i < 19; i += 4){
                                if(fabs(para[i+1]-para[i+5]) < 20){
                                    if(para[i] < para[i+4]){
                                        float dif = para[i+4] - para[i];
                                        para[i+4] -= dif;
                                        para[i] += dif;
                                    }
                                }
                            }
                        }
                        //generate hypo
                        MyHand.set_parameters(para);
                        MyHand.get_joints_positions();
                        //ros::Time time_begin = ros::Time::now();
                        //MyHand.get_handPointCloud(modelPointCloud);
                        MyHand.samplePointCloud(modelPointCloud);
                        //ros::Time time_end = ros::Time::now();
                        //std::cout<<"Duration: "<< time_end-time_begin << std::endl;
                        //******** 2.1 done ****************//

                        //******** 2.2 Ray tracing for Hypothesis ********//
                        Mat visiblityMap_Hypo(imageSize,imageSize,CV_8UC1,Scalar(back_groud_value));
                        pcl::PointCloud<pcl::PointXYZRGB> visibleModelPointCloud;
                        //Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo, visibleModelPointCloud);
                        Ray_tracing_OrthognalProjection(modelPointCloud, imageSize, resolution, visiblityMap_Hypo);
                        //******** 2.2 done *******************//

                        //        imshow("visibilityMap_Oberservation", visibilityMap_Oberservation);
                        //        imshow("visiblityMap_Hypo", visiblityMap_Hypo);
                        //        waitKey();

                        //ROS_INFO("Prepare Model Cloud");
                        //                        sensor_msgs::PointCloud2 model_cloud_msg;
                        //                        toROSMsg(visibleModelPointCloud,model_cloud_msg);
                        //                        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
                        //                        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
                        //                        modelPublisher_.publish(model_cloud_msg);

                        //******** 2.3 Score (similarity assessment) ******//
                        float overlap, overall_diff, overlap_diff, overlap_obs, overlap_hyp;
                        Score(visibilityMap_Oberservation, visiblityMap_Hypo, back_groud_value, overlap, overlap_obs, overlap_hyp, overall_diff, overlap_diff);
                        //******** 2.3 done *************//

                        //std::cout << "Overall_diff: " << overall_diff << std::endl;

                        if((overlap_obs/overlap <= Opt_Score_overlapratio1 || overlap_hyp/overlap <= Opt_Score_overlapratio2) && (overlap_diff/overlap <= Opt_Score_aver_overlapdiff || overlap_diff/overlap <= 1.1)){
                            Opt_Score_overlapratio1 = min(overlap_obs/overlap,Opt_Score_overlapratio1);
                            Opt_Score_overlapratio2 = min(overlap_hyp/overlap, Opt_Score_overlapratio2);
                            if(overlap_diff/overlap <= Opt_Score_aver_overlapdiff )
                                Opt_Score_aver_overlapdiff = 1.0*overlap_diff/overlap;
                            for(int i = 0; i< 27; i++){
                                optimal_para[i] = para[i];
                            }
                        }


                        for(int i = 0; i< 27; i++){
                            //std::cout << "para" << i <<": " << para[i] << std::endl;
                            para[i] = para_suboptimal[i];
                        }
                    }
                }
            }
            vector<float> for_the_que;
            for(int i = 0; i< 27; i++){
                std::cout << "para" << i <<": " << optimal_para[i] << std::endl;
                para_lastFrame[i] = optimal_para[i];
                for_the_que.push_back(optimal_para[i]);
            }
            parameter_que.add_new(for_the_que);
            parameter_que.smooth_mean(3);
            for(int i = 0; i< 27; i++){
                det_para[i] = parameter_que.para_delta[i];
            }

            std::cout << "Overlap ratio1: " <<Opt_Score_overlapratio1 << std::endl;
            std::cout << "Overlap ratio2: " <<Opt_Score_overlapratio2 << std::endl;
            std::cout << "Average overlap distance: " << Opt_Score_aver_overlapdiff << std::endl;

            //Gaution:
            //        boost::mt19937 *rng = new boost::mt19937();
            //        rng->seed(time(NULL));

            //        boost::normal_distribution<> distribution(70, 10);
            //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

            for(int i = 0; i< 27; i++){
                optimal_para[i] = parameter_que.para_sequence_smoothed[0][i];
            }
            MyHand.set_parameters(optimal_para);
            MyHand.get_joints_positions();
            //MyHand.get_handPointCloud(modelPointCloud);
            MyHand.samplePointCloud(modelPointCloud);
        }

        pcl::PointCloud<pcl::PointXYZRGB> articulation;
        for(int i = 0; i < 27; i++){
            articulation.push_back(MyHand.joints_position[i]);
        }






        //Gaution:
        //        boost::mt19937 *rng = new boost::mt19937();
        //        rng->seed(time(NULL));

        //        boost::normal_distribution<> distribution(70, 10);
        //        boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);





        seq++;
        if(seq > 10000)
            seq = 1;
        ///////////////////////////////////////////////////////////
        ros::Time Publish_begin = ros::Time::now();

        //ROS_INFO("Prepare Hand Cloud");
        sensor_msgs::PointCloud2 hand_cloud_msg;
        toROSMsg(handcloud,hand_cloud_msg);
        hand_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
        hand_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
        handPublisher_.publish(hand_cloud_msg);

        //ROS_INFO("Prepare Segmented Hand Cloud");
        sensor_msgs::PointCloud2 segmented_hand_cloud_msg;
        toROSMsg(Segmented_hand_cloud,segmented_hand_cloud_msg);
        segmented_hand_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
        segmented_hand_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
        segmented_hand_.publish(segmented_hand_cloud_msg);

        //ROS_INFO("Prepare Articulation");
        sensor_msgs::PointCloud2 cloud_msg_articulation;
        toROSMsg(articulation,cloud_msg_articulation);
        cloud_msg_articulation.header.frame_id=hand_kp_pter->header.frame_id;
        cloud_msg_articulation.header.stamp = hand_kp_pter->header.stamp;
        articulatePublisher_.publish(cloud_msg_articulation);

        //ROS_INFO("Prepare Model Cloud");
        sensor_msgs::PointCloud2 model_cloud_msg;
        toROSMsg(modelPointCloud,model_cloud_msg);
        model_cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
        model_cloud_msg.header.stamp = hand_kp_pter->header.stamp;
        modelPublisher_.publish(model_cloud_msg);

        //ROS_INFO("Prepare Hand Detection Result");
        visualization_msgs::Marker bone;
        bone.header.frame_id = hand_kp_pter->header.frame_id;
        bone.header.stamp = hand_kp_pter->header.stamp;
        bone.ns = "finger_tracking";
        bone.type = visualization_msgs::Marker::LINE_LIST;
        bone.id = 0;
        bone.action = visualization_msgs::Marker::ADD;
        bone.pose.orientation.w = 1.0;
        bone.scale.x = 0.001;
        bone.color.a = 1.0;
        bone.color.g = 1.0;
        for(int finger = 0; finger <5; finger++){
            for(int i = 1; i< 5; i++){
                geometry_msgs::Point p2;
                p2.x = articulation.points[5*finger+i].x;
                p2.y = articulation.points[5*finger+i].y;
                p2.z = articulation.points[5*finger+i].z;
                bone.points.push_back(p2);
                p2.x = articulation.points[5*finger+i+1].x;
                p2.y = articulation.points[5*finger+i+1].y;
                p2.z = articulation.points[5*finger+i+1].z;
                bone.points.push_back(p2);
            }
        }
        for(int i = 0; i< 2; i++){
            for(int j = 0; j< 3; j++){
                geometry_msgs::Point p2;
                p2.x = articulation.points[6+5*j+i].x;
                p2.y = articulation.points[6+5*j+i].y;
                p2.z = articulation.points[6+5*j+i].z;
                bone.points.push_back(p2);
                p2.x = articulation.points[6+5*j+5+i].x;
                p2.y = articulation.points[6+5*j+5+i].y;
                p2.z = articulation.points[6+5*j+5+i].z;
                bone.points.push_back(p2);
            }
        }
        geometry_msgs::Point p2;
        p2.x = articulation.points[1].x;
        p2.y = articulation.points[1].y;
        p2.z = articulation.points[1].z;
        bone.points.push_back(p2);
        p2.x = articulation.points[6].x;
        p2.y = articulation.points[6].y;
        p2.z = articulation.points[6].z;
        bone.points.push_back(p2);
        bone_pub_.publish( bone );

        ros::Time Publish_end = ros::Time::now();
        std::cout << Publish_end-Publish_begin << " seconds: Publish messages." << std::endl;










        ros::Time time9 = ros::Time::now();
        std::cout<< time9-time0 << "seconds: Total"<<std::endl;
        ROS_INFO("One callback done");


    }
    catch (std::exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}


