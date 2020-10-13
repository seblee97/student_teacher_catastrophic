#include <iostream>
#include <vector>
#include <string>

using namespace std;

class StudentTwoTeacherConfiguration {
    // overlap matrices
    vector<vector<float>> Q;
    vector<vector<float>> R;
    vector<vector<float>> U;
    vector<vector<float>> T;
    vector<vector<float>> S;
    vector<vector<float>> V;

    // student head weights
    vector<float> h1;
    vector<float> h2;

    // teacher head weights
    vector<float> th1;
    vector<float> th1;

    vector<vector<float>> global_covariance;

    public:
        vector<float> generate_covariance_matrix(vector<int>, int);
        void step_global_covariance();
        void step_overlap_matrix(string overlap_matrix_id, vector<vector<float>> matrix_delta);
        void step_head_weights(string head_weight_id, vector<float> weight_delta);
}

StudentTwoTeacherConfiguration::StudentTwoTeacherConfiguration (
    Q, R, U, T, S, V, h1, h2, th1, th2
) {
    Q=Q;
    R=R;
    U=U; 
    T=T;
    S=S;
    V=V;
    h1=h1;
    h2=h2;
    th1=th1;
    th2=th2;
}

StudentTwoTeacherConfiguration::generate_covariance_matrix (vector<int> indices, int teacher_index){
    vector<vector<float>> covariance;
    for (int i = 0; i < indices.size(); i++){
        for (int j = 0; j < indices.size(); j++){
            covariance[i][j] = this.global_covariance[i][j]
        }
    }
    return covariance;
}

StudentTwoTeacherConfiguration::step_global_covariance (){
    this.global_covariance = 
}

int main(){std::cout << "Hello World!";}