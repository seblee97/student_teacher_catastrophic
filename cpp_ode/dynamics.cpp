#include <string>

using namespace std;

class StudentTeacherODE {
    configuration overlap_configuration;
    string nonlinearity;
    float w_learning_rate;
    float h_learning_rate;
    float dt;
    bool soft_committee;
    bool train_first_layer;
    bool train_head_layer;

    public:
        void step();
}