#include <vector>

using namespace std;

class Overlap{
    vector<float> overlap_values;
    int timestep;
    bool final;

    public:
        void step();
};

class SelfOverlap: public Overlap {
   public:
      int getArea() { 
         return (width * height); 
      }
};