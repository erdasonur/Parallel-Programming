#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
using namespace std;

#define N 2000

double dMultiplicationArray[N][N] = { 0 };
double dMatrix1[N][N];
double dMatrix2[N][N];

float fMultiplicationArray[N][N] = { 0 };
float fMatrix1[N][N];
float fMatrix2[N][N];

void Block_Data_Sharing();
void Sequential_Data_Sharing();
void Multiplication_Serial_Double();
void Multiplication_Parallel_Double();
void Multiplication_Serial_Float();
void Multiplication_Parallel_Float();

int main()
{
	cout << "Wait please" << endl;
	clock_t start_time = clock(), end_time;
	//Block_Data_Sharing();
	//Sequential_Data_Sharing();
	//Multiplication_Serial_Double(); 
	Multiplication_Parallel_Double(); 
	//Multiplication_Serial_Float(); 
	//Multiplication_Parallel_Float(); 
	end_time = clock();
	cout << (float)(end_time - start_time) / CLOCKS_PER_SEC;
	return 0;
}

void Multiplication_Serial_Double() {

	int i, j, k;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			dMatrix1[i][j] = 1.0;
			dMatrix2[i][j] = 1.0;
		}
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			for (k = 0; k < N; k++)
			{
				dMultiplicationArray[i][j] += dMatrix1[i][k] * dMatrix2[k][j];
			}
		}
	}

	/*for (i= 0; i< N; i++)
	{
		for (j= 0; j< N; j++)
		{
			cout<<dMultiplicationArray[i][j]<<" ";
		}
		cout<<endl;
	}
	*/
	cout << "done ";
}

void Multiplication_Parallel_Double()
{
	int i, j, k;

#pragma omp parallel for collapse(2) private(i,j) shared(dMatrix1,dMatrix2) 
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++)
		{
			dMatrix1[i][j] = 1.0;
			dMatrix2[i][j] = 1.0;
		}
	}
#pragma omp parallel for collapse(2) private(i,j,k) shared(dMatrix1,dMatrix2,dMultiplicationArray)
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				dMultiplicationArray[i][j] += dMatrix1[i][k] * dMatrix2[k][j];
			}
		}
	}
	
	/*for (i= 0; i< N; i++)
	{
		for (j= 0; j< N; j++)
		{
			cout<<dMultiplicationArray[i][j]<<" ";
		}
		cout<<endl;
	}*/
	cout << "done ";
}
void Multiplication_Serial_Float() {

	int i, j, k;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			fMatrix1[i][j] = 1.0;
			fMatrix2[i][j] = 1.0;
		}
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				fMultiplicationArray[i][j] += fMatrix1[i][k] * fMatrix2[k][j];
			}
		}
	}

	/*for (i= 0; i< N; i++)
	{
		for (j= 0; j< N; j++)
		{
			cout<<fMultiplicationArray[i][j]<<" ";
		}
		cout<<endl;
	}
	*/
	cout << "done ";
}

void Multiplication_Parallel_Float()
{
	int i, j, k;
	

#pragma omp parallel for collapse(2) private(i,j) shared(fMatrix1,fMatrix2)
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			fMatrix1[i][j] = 1.0;
			fMatrix2[i][j] = 1.0;
		}
#pragma omp parallel for collapse(2) private(i,j,k) shared(fMatrix1,fMatrix2,fMultiplicationArray)
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				fMultiplicationArray[i][j] += fMatrix1[i][k] * fMatrix2[k][j];
			}
		}
	}
	/*#pragma omp parallel for private(i,j) shared(fMultiplicationArray)
	for (i= 0; i< N; i++)
	{
		for (j= 0; j< N; j++)
		{
			cout<<fMultiplicationArray[i][j]<<" ";
		}
		cout<<endl;
	}*/
	cout << "done ";
}

void Block_Data_Sharing() {

	int i, j, k;
#pragma omp parallel for collapse(2) private(i,j) shared(fMatrix1,fMatrix2)
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			fMatrix1[i][j] = 1.0;
			fMatrix2[i][j] = 1.0;
		}
#pragma omp parallel for  collapse(2) private(i,j,k)  shared(fMatrix1,fMatrix2,fMultiplicationArray) schedule(static,1) 
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				fMultiplicationArray[i][j] += fMatrix1[i][k] * fMatrix2[k][j];
			}
		}

	}
	/*for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			cout << fMultiplicationArray[i][j] << " ";
		}
		cout << endl;
	}*/
}

void Sequential_Data_Sharing() {

	int i, j, k;

#pragma omp parallel for collapse(2) private(i,j) shared(fMatrix1,fMatrix2)
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			fMatrix1[i][j] = 1.0;
			fMatrix2[i][j] = 1.0;
		}

#pragma omp parallel for collapse(2) private(j,i,k)  shared(fMatrix1,fMatrix2,fMultiplicationArray) schedule(static,1)
	for (j = 0; j < N; j++) {  
			for (i = 0; i < N; i++) {
				for (k = 0; k < N; k++) {
					fMultiplicationArray[i][j] += fMatrix1[i][k] * fMatrix2[k][j];
					//#pragma omp critical
					//cout << "i " << i << " j " << j << " tid " << omp_get_thread_num() << endl;
				}
				
			}
			
	}
	/*for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			cout << fMultiplicationArray[i][j] << " ";
		}
		cout << endl;
	}*/
}