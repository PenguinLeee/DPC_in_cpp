// Only the final matrix multiplication uses encryption. 
// Better use plaintext meessage in most of the space. 
// Since SVD blablabla will be used. 


#ifndef __DPC_H_
#define __DPC_H_

#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <list>
#include <ctime>
#include <random>

using namespace std;

static default_random_engine e(time(0));
static normal_distribution<double> white_noise(0,0.01); //white_noise

const int n = 4;
const int m = 1;
const int p = 1;

const int N = 20;
const int j = 1000;

const int num_lqr = 2 * N + j;
const int num_dpc = 200;

const double lambda = .005;

class Plant {
public:

    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd x;
    Eigen::MatrixXd x_plus;
    Eigen::MatrixXd y;
    Eigen::MatrixXd u;
    Eigen::MatrixXd r;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd K;

    Eigen::MatrixXd Up;
    Eigen::MatrixXd Uf;
    Eigen::MatrixXd Yp;
    Eigen::MatrixXd Yf;

    Eigen::MatrixXd Lwu;
    Eigen::MatrixXd Lw;
    Eigen::MatrixXd Lu;
    Eigen::MatrixXd W; // W = [Wp; Uf]
    Eigen::MatrixXd wp;
    Eigen::MatrixXd I;

    // for DOB design
    int DOB;
    Eigen::MatrixXd bh;
    Eigen::MatrixXd L;
    Eigen::MatrixXd u_com;
    Eigen::MatrixXd Lw_dob;
    Eigen::MatrixXd Lu_dob;

    Plant(int _dob);
    ~Plant();

    //historical data
    Eigen::MatrixXd us;
    Eigen::MatrixXd ys;
    

    void generateHankel();
    void updateHankel();

    void step(Eigen::MatrixXd& _u);
    void stepLQR();
    void recorduy();
    void makeupW();
    void DPCstep();

    void dob();
}; 

Plant::Plant(int _dob) {
    DOB = _dob;
    A = Eigen::MatrixXd::Random(n, n);
    B = Eigen::MatrixXd::Random(n, m);
    C = Eigen::MatrixXd::Random(p, n);
    x = Eigen::MatrixXd::Random(n, 1);
    x_plus = Eigen::MatrixXd::Random(n, 1);
    y = Eigen::MatrixXd::Random(p, 1);
    u = Eigen::MatrixXd::Random(m ,1);
    r = Eigen::MatrixXd::Random(p, 1);

    Q = Eigen::MatrixXd::Random(n, n);
    R = Eigen::MatrixXd::Random(m, m);
    K = Eigen::MatrixXd::Random(m, n);

    // us and ys for generating & updating hankel matrix. 
    us = Eigen::MatrixXd::Random((2*N+j-1)*m, 1);
    ys = Eigen::MatrixXd::Random((2*N+j-1)*p, 1);

    // Hankel Matrix
    // Wp: [N(p+m), j]
    Up = Eigen::MatrixXd::Random(N*m, j);
    Uf = Eigen::MatrixXd::Random(N*m, j);
    Yp = Eigen::MatrixXd::Random(N*p, j);
    Yf = Eigen::MatrixXd::Random(N*p, j);

    // Coeff
    Lwu = Eigen::MatrixXd::Random(N*p, N*(p+2*m));
    Lw = Eigen::MatrixXd::Random(N*p, N*(p+m));
    Lu = Eigen::MatrixXd::Random(N*p, N*m);
    W = Eigen::MatrixXd::Random(N*(2*m+p),j); // W = [Wp; Uf]
    wp = Eigen::MatrixXd::Random(N*(m+p),1);
    I = Eigen::MatrixXd::Identity(N*m, N*m);

    //DOB
    L = Eigen::MatrixXd::Random(m, p);
    u_com = Eigen::MatrixXd::Random(m ,1);
    Lw_dob = Eigen::MatrixXd::Random(p, Lw.cols());
    Lu_dob = Eigen::MatrixXd::Random(m, Lu.cols());

    A << 0.9932, 0.0029, 0.0, 0.0036, 0.0012, 0.9945, 0.0004, 0.0036, 0.0, 0.0004, 0.9769, 0.0, 0.2305, 0.5577, 0.0001, 0.1503;
    B << 0.0003, 0.0003, 0.0, 0.0579;
    C << 0.0,0.0,0.0,1.0;
    K << -0.26055977, -0.52474278, -0.00820757, -0.0120031;
    r << 18.0;

    Q << 1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0;
    R << 1.0;

    // init x
    x << 19.9957977734295334, 19.9794826103046823, 9.1384167046755778, 20.6402718884586314;

    y = C * x;
    
    L << -10;
}

Plant::~Plant() {}

void Plant::step(Eigen::MatrixXd& u) {
    
    Eigen::MatrixXd noise_x = Eigen::MatrixXd::Zero(n,1).unaryExpr([](double dummy){return white_noise(e);});
    Eigen::MatrixXd noise_y = Eigen::MatrixXd::Zero(p,1).unaryExpr([](double dummy){return white_noise(e);});
    x_plus = A * x + B * u + noise_x;
    y = C * x_plus + noise_y;
    x = x_plus;
    // cout << "A: " << A.rows() << " by " << A.cols() << endl;
    // cout << "After updating state. " << endl;
    cout << y << endl;
    recorduy();
}

void Plant::stepLQR() {
    // cout << "In stepLQR" << endl;
    u = K * x;
    // cout << "after computing u" << endl; 
    step(u);
}

void Plant::generateHankel() {
    // int rowsUp = Up.rows();
    for(int idx = 0; idx < j ;idx ++) {
        Up.block(0, idx, N*m, 1) = us.block(idx*m, 0, N*m, 1);
    }

    // int rowsUf = Uf.rows();
    for(int idx = 0; idx < j ;idx ++) {
        Uf.block(0, idx, N*m, 1) = us.block((idx+N)*m, 0, N*m, 1);
    }

    // int rowsYp = Yp.rows();
    for(int idx = 0; idx < j ;idx ++) {
        Yp.block(0, idx, N*p, 1) = ys.block(idx*p, 0, N*p, 1);
    }

    // int rowsYf = Yf.rows();
    for(int idx = 0; idx < j ;idx ++) {
        Yf.block(0, idx, N*p, 1) = ys.block((idx+N)*p, 0, N*p, 1);
    }
}

void Plant::recorduy() {
    // cout << us.rows() << " by " << us.cols() << endl;
    us.block(0, 0, (2*n+j-2)*m, 1) = us.block(m, 0, (2*n+j-2)*m, 1);
    us.block((2*n+j-2)*m, 0, m, 1) = u.block(0,0,m,1);

    ys.block(0, 0, (2*n+j-2)*p, 1) = us.block(p, 0, (2*n+j-2)*p, 1);
    ys.block((2*n+j-2)*p, 0, p, 1) = y.block(0,0,p,1);
}

void Plant::updateHankel() {

    Up.block(0, 0, Up.rows(), Up.cols()-2) = Up.block(0, 1, Up.rows(), Up.cols()-2);
    Up.block(0, Up.cols()-1, Up.rows(), 1) = us.block((j-1)*m, 0, Up.rows(), 1);

    Uf.block(0, 0, Uf.rows(), Uf.cols()-2) = Uf.block(0, 1, Uf.rows(), Uf.cols()-2);
    Uf.block(0, Uf.cols()-1, Uf.rows(), 1) = us.block((N+j-1)*m, 0, Uf.rows(), 1);

    Yp.block(0, 0, Yp.rows(), Yp.cols()-2) = Yp.block(0, 1, Yp.rows(), Yp.cols()-2);
    Yp.block(0, Yp.cols()-1, Yp.rows(), 1) = ys.block((j-1)*p, 0, Yp.rows(), 1);

    Yf.block(0, 0, Yf.rows(), Yf.cols()-2) = Yf.block(0, 1, Yf.rows(), Yf.cols()-2);
    Yf.block(0, Yf.cols()-1, Yf.rows(), 1) = ys.block((N+j-1)*p, 0, Yf.rows(), 1);
}

void Plant::makeupW() {
    W.block(0, 0, Yp.rows(), Yp.cols()) = Yp.block(0, 0, Yp.rows(), Yp.cols());
    W.block(Yp.rows(), 0, Up.rows(), Up.cols()) = Up.block(0, 0, Up.rows(), Up.cols());
    W.block(Yp.rows()+Up.rows(), 0, Uf.rows(), Uf.cols()) = Uf.block(0, 0, Uf.rows(), Uf.cols());
    wp = W.block(0, W.cols()-1, wp.rows(), wp.cols());
}

void Plant::DPCstep() {
    makeupW();
    Lwu = W.transpose().completeOrthogonalDecomposition().solve(Yf.transpose()).transpose();
    Lw = Lwu.block(0, 0, Lw.rows(), Lw.cols());
    Lu = Lwu.block(0, Lw.cols(), Lu.rows(), Lu.cols());
    Eigen::MatrixXd uf = (lambda * I + Lu.transpose()*Lu).inverse()*Lu.transpose()*(r.replicate(N, 1) - Lw*wp);
    u = uf.block(0, 0, u.rows(), u.cols());
    if(DOB == 1)
        dob();
    step(u);
    updateHankel();
}

void Plant::dob() {
    Eigen::MatrixXd u_tmp = us.block((N+j-1)*m, 0, N*m, 1); // (2*N+j-1)*m, 1
    Eigen::MatrixXd y_tmp = ys.block((N+j-1)*p, 0, N*p, 1);
    Eigen::MatrixXd w(N*(m+p), 1); 
    w << u_tmp, y_tmp;
    Lw_dob = Lw.block(0, 0, Lw_dob.rows(), Lw_dob.cols());
    Lu_dob = Lu.block(0, 0, Lu_dob.rows(), m);

    u_com = (-1) * L * (y - Lw_dob * w + Lu_dob * u);
    u = u - u_com;
}

#endif