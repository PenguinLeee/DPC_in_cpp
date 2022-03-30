#include <iostream>
#include "dpc.h"

#include <Eigen/Eigen>

using namespace std;

void data_driven_control();

int main() {
    data_driven_control();
    return 0;
}

void data_driven_control() {
    Plant plant = Plant(1);
    // cout << "A: we call " << plant.A.rows() << " by " << plant.A.cols() << endl;
    for(int index = 0; index < num_lqr; index++) {
        cout << "LQR iter " << index << endl;
        plant.stepLQR();
    }
    cout << "LQR data generation completed. " << endl << endl;
    
    cout << "DPC starts. " << endl;
    plant.generateHankel();

    for(int index = 0; index < num_dpc; index++) {
        // make up W = [Yp; Up; Uf]
        plant.DPCstep();
    }
    cout << "All finished!" << endl;
}