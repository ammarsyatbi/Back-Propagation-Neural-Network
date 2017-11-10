#include <iostream>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <windows.h>

using namespace std;

void readData(struct BPP &bpp);
void initializeWeight(struct BPP &bpp);
void workout(struct BPP &bpp);
void step4(struct BPP &bpp,int instances);
void step5(struct BPP &bpp,int instances);
void step6(struct BPP &bpp,int instances);
void step7(struct BPP &bpp,int instances);
void step8(struct BPP &bpp);
void calculateMSE(struct BPP &bpp);
void printNewWeight(struct BPP bpp);
void report(struct BPP bpp);
void saveWeight(struct BPP bpp);
void printOutput(struct BPP bpp);
void saveOutput(struct BPP bpp);
double calculateAccuracy(struct BPP &bpp);
void saveOldWeight(struct BPP bpp);

const int MAX_EPOCH = 100000;
const double LEARNING_RATE = 0.5;
const int NUM_OF_INSTANCES = 305;
const double MIN_MSE = 0.0001;
const int SP = 4; //set precision.

const int NUM_OF_INPUT = 6;
const int NUM_OF_HIDDEN = 5;
const int NUM_OF_OUTPUT = 1;//tak payah takpe

struct BPP{

    //input
    double x[NUM_OF_INSTANCES][NUM_OF_INPUT];

    //output / traget
    double target_y[NUM_OF_INSTANCES];//sebab tak 2 array weh
    double y[NUM_OF_INSTANCES];
    double y_in;

    //hidden bebeh
    double z[NUM_OF_HIDDEN];
    double z_in[NUM_OF_HIDDEN];


    //weights
    //v[i][j]
    //row i , col j
    double v[6][5];
    double v_chg[6][NUM_OF_HIDDEN];//wek ceng
    //wjk
    //row j, col k :: row satu je pfft
    double w[NUM_OF_HIDDEN];
    double w_chg[NUM_OF_HIDDEN];//wek ceng

    //mean square error
    double MSE;

    double deltaK;
    double deltaJin[NUM_OF_HIDDEN];
    double deltaJ[NUM_OF_HIDDEN];
    int epo;
}bpp;

int main()
{

    initializeWeight(bpp);//initialize weight
    saveOldWeight(bpp);
    readData(bpp);//read data from text
    printNewWeight(bpp);
    //training kasi sado
    workout(bpp);
    printNewWeight(bpp);
    printOutput(bpp);
    report(bpp);
    cout << "accuracy : " << calculateAccuracy(bpp);
    saveWeight(bpp);
    saveOutput(bpp);
    return 0;
}
void initializeWeight(struct BPP &bpp)
{
    // the same loops for cal weight
    for(int r = 0; r<6; r++)
    {
        for(int c =0; c <NUM_OF_HIDDEN; c++)
        {

            bpp.v[r][c] = (((float)rand()/ RAND_MAX)*2.4)/35;
        }
    }

    for(int i =0; i <NUM_OF_HIDDEN; i ++)
    {

        bpp.w[i] = (((float)rand()/ RAND_MAX)*2.4)/35;
    }
}
void readData(struct BPP &bpp)
{

    ifstream vetebral;
    vetebral.open("tab.txt");


    //cout << setw(5) <<  "x1" << setw(10) <<  "x2" << setw(10) <<  "x3" << setw(10) <<  "x4" << setw(10) <<  "x5" << setw(10) <<  "x6" << setw(10) <<  "y1" << endl;

    for(int i =0; i <NUM_OF_INSTANCES; i++)
    {
        //in is number of X
        for(int in = 0; in <NUM_OF_INPUT; in++)
        {

            vetebral >> bpp.x[i][in];
            //cout << setprecision(4) << setw(10) <<  bpp.x[i][in];
        }
         vetebral >> bpp.target_y[i];
        //cout << setprecision(4) << setw(10) <<  bpp.target_y[i] << endl;
        //sebab nanti output tak cantik
        //cout << setprecision(4) << setw(10) <<  bpp.x[i][0]<< setw(10) <<  bpp.x[i][1] << setw(10) <<  bpp.x[i][2]<< setw(10) <<  bpp.x[i][3]<< setw(10) ;
        //cout <<  bpp.x[i][4]<< setw(10) <<  bpp.x[i][5]<<  setw(10) <<  bpp.y1[i] << endl;

    }

    vetebral.close();

}
void workout(struct BPP &bpp)
{
    bpp.MSE = 1;//nak bagi masuk loop je
    int epoch =0;
    while((epoch < MAX_EPOCH) && (bpp.MSE > MIN_MSE))
    {

        for(int instances = 0; instances < NUM_OF_INSTANCES; instances++)
        {
            //foward , one flow , one instances per iteration
            step4(bpp,instances);
            step5(bpp,instances);
            //backward, dah start nak tukar weight jeng jeng jeng
            step6(bpp,instances);
            step7(bpp,instances);
            //hok ni update weight for next instances
            step8(bpp);
            //cout << endl;
            //printNewWeight(bpp);
            //system("PAUSE");
             //nak test je

        }
        bpp.epo = epoch;
        //printOutput(bpp);
        calculateMSE(bpp);//penentu epoc hewhewhew
        report(bpp);

        epoch++;
    }

    report(bpp);

}
void step4(struct BPP &bpp,int instances)
{
    //calculate zj in,

    for(int r = 0; r<NUM_OF_HIDDEN; r++)
    {
        bpp.z_in[r] = 0;
        for(int c =0; c <6; c++)
        {
           bpp.z_in[r] += bpp.x[instances][c] * bpp.v[c][r];//zjin
        }
        bpp.z[r] = 1 / (1 + exp(-bpp.z_in[r]));
    }

}
void step5(struct BPP &bpp,int instances)
{
    bpp.y_in = 0;

     for(int c = 0; c<NUM_OF_HIDDEN; c++)
    {
        bpp.y_in += bpp.z[c] * bpp.w[c];//zjin
    }

    bpp.y[instances] = 1 / (1 + exp(-bpp.y_in));
}

void step6(struct BPP &bpp,int instances)
{
    bpp.deltaK = 0;//k ade satu je pon

    bpp.deltaK = (bpp.target_y[instances] - bpp.y[instances]) *(bpp.y[instances]) * (1 - bpp.y[instances]);

     for(int c = 0; c<NUM_OF_HIDDEN; c++)
    {
        bpp.w_chg[c] = LEARNING_RATE * bpp.deltaK * bpp.z[c];//zjin //w11 w21 w31 w41 w51
         //cout << bpp.w_chg[c] << endl;

    }
}

void step7(struct BPP &bpp,int instances)
{
    //cal deltaJ in;
    //double deltaJin[5];
    //double deltaJ[5];

    //kalau ikut formula dia kene sum K , tapi sebab K ade 1 je hewhewhwe takyah
    for(int j =0; j <NUM_OF_HIDDEN; j++)
    {
        bpp.deltaJin[j] = bpp.deltaK * bpp.w[j]; //k takyah sebab 1 !!!!!!!!demyu bren
    }
    for(int j =0; j <NUM_OF_HIDDEN; j++)
    {
        bpp.deltaJ[j] = bpp.z[j] * (1 - bpp.z[j]); //k takyah sebab 1 !!!!!!!!demyu bren
    }
    //wek ceng for vij
    for(int j = 0; j<NUM_OF_HIDDEN; j++)
    {
        for(int i =0 ; i<6; i++)
        {
            bpp.v_chg[i][j] = LEARNING_RATE * bpp.deltaJ[j] * bpp.x[instances][i];
            //cout << bpp.v_chg[i][j] << endl;
        }
    }
}
void step8(struct BPP &bpp)
{

    //update Wij
    for(int i = 0 ; i < 6; i++)
    {
        for(int j = 0; j <NUM_OF_HIDDEN; j++)
        {
            bpp.v[i][j] = bpp.v[i][j] + bpp.v_chg[i][j];
        }
    }

    //update Vij
    //kenapa tak dua loop? SEBAB K ADE SATU JE
    for(int j = 0; j < 5; j++)
    {
        bpp.w[j] = bpp.w[j] + bpp.w_chg[j];
    }
}
void calculateMSE(struct BPP &bpp)
{
    double sumMSE =0;
    double error ;
    for(int i =0; i < NUM_OF_INSTANCES; i++)
    {
        //cout << " TARGET : " << bpp.target_y[i]  << " U : " << bpp.y[i];;
        error = bpp.target_y[i] - bpp.y[i];
        //cout << " ERROR : " << error;
        sumMSE += pow(error,2);
    }
    bpp.MSE = sumMSE / NUM_OF_INSTANCES;
}
void printNewWeight(struct BPP bpp)
{


        //update Wij
    for(int i = 0 ; i < 6; i++)
    {
        for(int j = 0; j <NUM_OF_HIDDEN; j++)
        {
            cout << setprecision(SP) << setw(10) << bpp.v[i][j] ;
        }
        cout << endl;
    }

    //update Vij
    //kenapa tak dua loop? SEBAB K ADE SATU JE
     cout << endl;
    for(int j = 0; j < NUM_OF_HIDDEN; j++)
    {
        cout << setprecision(SP) << setw(10) <<bpp.w[j];
    }
    cout << endl;
}

void saveWeight(struct BPP bpp)
{
    ofstream saver;
    saver.open("newWeight.txt");

    for(int i = 0 ; i < 6; i++)
    {
        for(int j = 0; j <NUM_OF_HIDDEN; j++)
        {
            saver << setprecision(SP) << bpp.v[i][j] << " ";
        }
        saver << endl;
    }

        for(int j = 0; j < 5; j++)
    {
        saver << setprecision(SP) <<bpp.w[j] << " ";
    }
    saver << endl;

    saver.close();
}
void saveOldWeight(struct BPP bpp)
{
    ofstream saver;
    saver.open("oldWeight.txt");

    for(int i = 0 ; i < 6; i++)
    {
        for(int j = 0; j <NUM_OF_HIDDEN; j++)
        {
            saver << setprecision(SP) << bpp.v[i][j] << " ";
        }
        saver << endl;
    }

        for(int j = 0; j < 5; j++)
    {
        saver << setprecision(SP) <<bpp.w[j] << " ";
    }
    saver << endl;

    saver.close();
}
void report(struct BPP bpp)
{
    cout << endl;
    cout << setw(10) << "MSE : "  << bpp.MSE << endl;
    cout << setw(10) << "EPOCH : " << bpp.epo << endl;
    cout << endl;

}
void printOutput(struct BPP bpp)
{
    cout << setw(10) <<  "Output" << setw(10) <<  "Target" << endl;

    for(int i =0; i <NUM_OF_INSTANCES; i++)
    {
        //in is number of X

        cout << setprecision(4) << setw(10)<<  bpp.y[i] << setw(10)<<  bpp.target_y[i] << endl;
        //sebab nanti output tak cantik
        //cout << setprecision(4) << setw(10) <<  bpp.x[i][0]<< setw(10) <<  bpp.x[i][1] << setw(10) <<  bpp.x[i][2]<< setw(10) <<  bpp.x[i][3]<< setw(10) ;
        //cout <<  bpp.x[i][4]<< setw(10) <<  bpp.x[i][5]<<  setw(10) <<  bpp.y1[i] << endl;

    }

}
void saveOutput(struct BPP bpp)
{
    ofstream out;
    out.open("output.txt");
    out << setw(10) <<  "x1" << setw(10) <<  "x2"<< setw(10) <<  "x3"<< setw(10) <<  "x4"<< setw(10) <<  "x5"<< setw(10) <<  "x6" << setw(10) <<  "Output" << setw(10) <<  "Target" << endl;

    for(int i =0; i <NUM_OF_INSTANCES; i++)
    {
        //in is number of X

        out << setprecision(4)<< setw(10)<<  bpp.x[i][0]<< setw(10)<< bpp.x[i][1] << setw(10)<<  bpp.x[i][2];
        out << setprecision(4) << setw(10)<<  bpp.x[i][3]<< setw(10)<<  bpp.x[i][4]<< setw(10)<<  bpp.x[i][5] << setw(10)<<  bpp.y[i] << setw(10)<<  bpp.target_y[i] << endl;
        //sebab nanti output tak cantik
        //cout << setprecision(4) << setw(10) <<  bpp.x[i][0]<< setw(10) <<  bpp.x[i][1] << setw(10) <<  bpp.x[i][2]<< setw(10) <<  bpp.x[i][3]<< setw(10) ;
        //cout <<  bpp.x[i][4]<< setw(10) <<  bpp.x[i][5]<<  setw(10) <<  bpp.y1[i] << endl;

    }
    out.close();
}
double calculateAccuracy(struct BPP &bpp)
{

    double sum =0;
    double q;
    int x;
    for(int i =0; i <NUM_OF_INSTANCES; i++)
    {
        q =  bpp.target_y[i] - bpp.y[i];
        if(q<0){q=q*(-1);}

      if(q<0.1)
      {
          sum += 1;
      }

    }

    double total = sum / NUM_OF_INSTANCES;

    return total;
}
