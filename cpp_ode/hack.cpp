#include <iostream>
#include <vector>
#include <string>

using namespace std;

vector< vector<int> > transpose (vector< vector<int> > array){
    int x_dimension = array[0].size();
    int y_dimension = array.size();

    vector< vector<int> > transposed_array;

    for (int i = 0; i < x_dimension; i++){
        for (int j = 0; j < y_dimension; j++){
            transposed_array[j][i] = array[i][j];
        }
    }
    return transposed_array;
}

void print_array(vector< vector<int> > array){
    for ( int i = 0; i < array.size(); i++ )
    {
        for ( int j = 0; j < array[i].size(); j++ ) cout << array[i][j] << ' ';
        cout << endl;
    }
}

int main(){
    vector< vector<int> > a = { { 1, 2, 3 }, { 2, 2, 2 } };
    vector< vector<int> > b = { { 4, 5, 6 }, { 6, 6, 6 } };
    vector< vector<int> > c;
    vector< vector<int> > a_transpose;
    vector< vector<int> > b_transpose;
    vector< vector<int> > d;

    c = a;

    a_transpose = transpose(a);
    // b_transpose = transpose(b);

    // d = a_transpose;

    c.insert( c.end(), b.begin(), b.end() );
    // d.insert( d.end(), b_transpose.begin(), b_transpose.end() );

    print_array(c);
    // print_array(transpose(d));
}