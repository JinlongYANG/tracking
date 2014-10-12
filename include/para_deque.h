#ifndef para_deque_h
#define para_deque_h

#include <iostream>


using namespace cv;

vector<float> vector_add(const vector<float> a, const vector<float> b){
    vector<float> addition_result;
    if(a.size() == b.size()){
        for(int i = 0; i< a.size(); i++){
            addition_result.push_back(a[i] + b[i]);
        }
        return addition_result;
    }
    else{
        ROS_ERROR("Different vector size in addition!");
        return addition_result;
    }
}

vector<float> vector_add(const vector<float> a, const vector<float> b, const vector<float> c){
    vector<float> addition_result = vector_add(a,vector_add(b,c));
    return addition_result;

}

vector<float> vector_weighted_add(const vector<float> a, const float weight_a, const vector<float> b){
    vector<float> addition_result;
    if(a.size() == b.size()){
        float weight_b = 1-weight_a;
        for(int i = 0; i< a.size(); i++){
            addition_result.push_back(weight_a* a[i] + weight_b*b[i]);
        }
        return addition_result;
    }
    else{
        ROS_ERROR("Different vector size in addition!");
        return addition_result;
    }
}

vector<float> vector_weighted_add(const vector<float> a, const float weight_a, const vector<float> b, const float weight_b, const vector<float> c){
    vector<float> addition_result;
    if(a.size() == b.size() && b.size() == c.size()){
        float weight_c = 1-weight_a - weight_b;
        for(int i = 0; i< a.size(); i++){
            addition_result.push_back(weight_a*a[i] + weight_b*b[i] + weight_c*c[i]);
        }
        return addition_result;
    }
    else{
        ROS_ERROR("Different vector size in addition!");
        return addition_result;
    }
}

vector<float> vector_subtrac(const vector<float> a, const vector<float> b){
    vector<float> subtract_result;
    if(a.size() == b.size()){
        for(int i = 0; i< a.size(); i++){
            subtract_result.push_back(a[i] - b[i]);
        }
        return subtract_result;
    }
    else{
        ROS_ERROR("Different vector size in subtraction!");
        return subtract_result;
    }
}

class para_deque
{
public:

    std::deque< vector<float> > para_sequence;
    std::deque< vector<float> > para_sequence_smoothed;
    vector<float> para_delta;
    int size_;
    int max_size_;

    para_deque(const int maximum_size = 7){
        size_ = 0;
        max_size_ = maximum_size;
        for( int i = 0; i< 27; i++){
            para_delta.push_back(0);
        }
    }

    void add_new(const vector <float> new_para){
        if(new_para.size() == 27){
            if(para_sequence.size() != max_size_){
                para_sequence.push_front(new_para);
                size_ ++;
            }
            else{
                para_sequence.pop_back();
                para_sequence.push_front(new_para);
            }
        }
        std::cout << "Queue size: " << para_sequence.size() << std::endl;
//        for(int i=0; i<para_sequence[0].size(); i++ ){
//            std::cout << "Para " << i <<": " << para_sequence[0][i] << std::endl;
//        }
    }

    void smooth_mean(const int window_size){
        if(window_size != 2 && window_size != 3){
            ROS_ERROR("Wrong window size, can not smooth the parameter queue!");
        }
        else{
            if(window_size == 2){
                if(size_ < 2){
                    if(size_ == 1)
                        para_sequence_smoothed.push_front(para_sequence[0]);
                }
                else {
                    float weight = 0.6;
                    vector<float> temp = vector_weighted_add(para_sequence[0], weight, para_sequence[1]);
                    if(para_sequence_smoothed.size() != max_size_)
                        para_sequence_smoothed.push_front(temp);
                    else{
                        para_sequence_smoothed.pop_back();
                        para_sequence_smoothed.push_front(temp);
                    }
                    para_delta = vector_subtrac(para_sequence_smoothed[0], para_sequence_smoothed[1]);
                }
            }
            else if (window_size == 3){
                if(size_ < 3){
                    if(size_ == 1)
                        para_sequence_smoothed.push_front(para_sequence[0]);
                    else if(size_ == 2){
                        para_sequence_smoothed.push_front(para_sequence[0]);
                    }
                }
                else{
                    float weight_a = 0.3, weight_b = 0.4;
                    vector<float> temp = vector_weighted_add(para_sequence[0], weight_a, para_sequence[1], weight_b, para_sequence[2]);
                    if(para_sequence_smoothed.size() != max_size_){
                        para_sequence_smoothed.pop_front();
                        para_sequence_smoothed.push_front(temp);
                        para_sequence_smoothed.push_front(vector_weighted_add(para_sequence[0], 0.7, para_sequence_smoothed[0]));
                    }
                    else{
                        para_sequence_smoothed.pop_front();
                        para_sequence_smoothed.push_front(temp);
                        para_sequence_smoothed.push_front(vector_weighted_add(para_sequence[0], 0.7, para_sequence_smoothed[0]));
                    }
                    para_delta = vector_subtrac(para_sequence_smoothed[0], para_sequence_smoothed[1]);
                }
            }

        }
    }

private:


};

#endif // para_deque.h

