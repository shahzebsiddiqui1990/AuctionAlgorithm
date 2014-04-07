/* Name: Shahzeb Siddiqui
*  Email: shahzeb.siddiqui@us.ibm.com
*  Program: Auction Algorithm
*  Description: This program implements the auction algorithm which is 
*  used for solving the assignment problem. In auction, there are 
*  buyers, and items that need to be matched such that the maximum  
*  cardinality and sum across diagonal is obtained. The input file 
*  contains x,y coordinate and the nonzero values in the matrix. The 
*  matrix is transformed into a Compressed Sparse Row format for better 
*  memory usage. The algorithm utilizes price vector for keeping track 
*  of item's price during auction to correctly assign items to the 
*  highest buyer.
*

/* LIBRARIES */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include "timer.h"

/* MACRO DEFINITION */
#define AUCTION_SETUP_TIMING
//#define AUCTION_TIMING_INFO
//#define SHOW_PREAUCTION_INFO
//#define AUCTION_SUMMARY
//#define SORT_BUYERS_ITEM

#define _MALLOC(P,T,S) {P=(T*)malloc((S)*sizeof(T));assert(P!= NULL);}
#define _FREE(P) { if ( (P) != NULL) free(P); }
  
/* 				Buyer Struct Details
 * biditemsvalue: list of nonzeros in buyer's row
 * biditemsindex: list of nonzeros index (column) in buyer's row
 * firstprofit: highest profit defined as j1 = max(aij - pj)
 * firstprofitindex: item id for firstprofit
 * secondprofit: second highest profit defined as j2 = max(aij - pj) where j2 != j1
 * secondprofitindex: item if for secondprofit
 * numitems: number of items (# of nonzeros) buyer can bid
 * buyer_rowid: buyer's reference  in terms of row # in matrix
 * matched: matched = 1 buyer is matched to item, otherwise 0
 */
typedef struct 
{
  float *biditemsvalue;							
  float firstprofit;
  float secondprofit;
  int firstprofitindex;
  int secondprofitindex;
  int *biditemsindex;								
  int numitems;									
  int buyer_rowid;								
  int matched;															
}buyer;
	

/*				Item Struct Details
 * items_bidder: list of buyers for each item
 * numbidders: number of nonzeros in column (represent number of buyers)
 * matched: matched = 1 item is matched to buyer, otherwise 0
 */ 
typedef struct 
{	
  int *items_bidder;								
  int numbidders;									
  int matched;										
}item;

/* matrix parameters */
int matrix_row, matrix_col, matrix_nonzeros;

/* array of buyer structs that representating each row in matrix */
buyer *listofbuyers;

/* array of item structs that represent each column in matrix */
item *listofitems;

/* array holding diagonal entries in auction to produce maximum sum */
float *diagonalmax;

/* matching matrix for diagonal */
int *M;

/* Compressed Sparse Row Format arrays */
float *valueptr, *origvalue;
int *colptr;
int *rowptr;

/* array keeps track of all items while reading from file  */
int *unique_item;

/* array tracking the number of bidders for each item indexed in order as items 
 * present in array "unique_item" */
int *bidder_per_item;

/* gmin, gmax finds absolute  min and max value in entire matrix,
 * delta = log(gmin/gmax)*/
float gmin, gmax, delta;
float eps;
float *pricevector;
/* array holding the maximum value of each row which is used for 
 * normalizing weights*/
float *mi;

// minimum value for GMIN  
double GMIN_THRESHOLD;


/* FUNCTION DEFINITIONS */
void ReadMatrix(char *filename);
void AuctionSetup(float *diagonalcost);
void Auction(float diagonalcost);
void PrintList();
void Summary();
float GetMaxRowValue(float *rowlist, int numitems);
void SortBuyersItemGreatestToLeast(int *sort_item_index, float *sort_item_value,
                int *item_index, float *item_value, int numitems);
void shuffle(int *array, size_t n);

int main(int argc, char ** argv)
{
  float diagonalcost;
  GMIN_THRESHOLD = 10e-35;

  StartTimer();
  ReadMatrix(argv[1]);   
  printf("Time to Pre-processing and Reading  [sec]:  %f\n", 
         GetTimer()/1000.f);
  
  StartTimer();
  AuctionSetup(&diagonalcost);
  printf("Time to Setup Auction [sec]:	%f\n", GetTimer()/1000.f);
  
  StartTimer();
  Auction(diagonalcost);
  printf("Time for Auction 	[sec]:	%f\n", GetTimer()/1000.f);
  
  #ifdef AUCTION_SUMMARY
  Summary();
  #endif
  
//  PrintList();
  return 0;
}


/********************************************************************
 * Auction Setup
 * 
 * Description: This function initializes and assigns metadata for 
 * buyer and item struct which will be used for Auction algorithm. 
 * Buyers are represented as rows in matrix and each buyer contains a 
 * list of nonzero entries in its row which represents item that buyer 
 * can bid. Similarly, each item (column in matrix) can be bid by 
 * multiple buyers if there are 2 or more nonzeros in column. The 
 * weights are normalized for numerical stability based on 
 * Anshul/Yinglong's paper.  									
 * *****************************************************************/
void AuctionSetup (float *diagonalcost)
{
  int i,j;
	
  // Initializing listofbuyers and listofitems  and its metadata
  _MALLOC(listofbuyers,buyer,matrix_row);
  _MALLOC(listofitems,item,matrix_col);
  
  _MALLOC(M,int,matrix_row);
  
  _MALLOC(diagonalmax,float,matrix_row);
  _MALLOC(pricevector,float,matrix_row);
  // setting diagonal and pricevector to 0, and Matching vector to -1
  for (i = 0; i < matrix_row; i++)
  {
    diagonalmax[i] = 0;
	pricevector[i] = 0;
	M[i] = -1;
  }	

#ifdef AUCTION_SETUP_TIMING
  double t1;
  StartTimer();
#endif
	
  for (i = 0; i < matrix_row; i++)
  {
	// buyer struct setup
	listofbuyers[i].numitems = 0;
	listofbuyers[i].matched = 0;
	listofbuyers[i].buyer_rowid = i;	
	listofbuyers[i].firstprofit = listofbuyers[i].secondprofit = 0;
	listofbuyers[i].firstprofitindex = listofbuyers[i].secondprofitindex = -1; 
		
	// item struct setup
	listofitems[i].numbidders = 0;					
	listofitems[i].matched = 0;		
  }
	
#ifdef AUCTION_SETUP_TIMING
  t1 = GetTimer()/1000.f;
  printf("Auction Setup Allocation Time [sec]:	%f\n", t1);
  StartTimer();
#endif
	
// acquire number of items for each buyer and assign item's list for each buyer
  for (i = 0; i < matrix_row; i++)
  {		
	listofbuyers[i].numitems = rowptr[i+1] - rowptr[i];
			
	// biditemsindex store column index for all nonzeros in buyer's row
	_MALLOC(listofbuyers[i].biditemsindex,int,listofbuyers[i].numitems);		  
		
	// biditemsvalue store nonzero value for each buyer's row
	_MALLOC(listofbuyers[i].biditemsvalue,float,listofbuyers[i].numitems);					
	
	memset(listofbuyers[i].biditemsindex,0,listofbuyers[i].numitems);
	memset(listofbuyers[i].biditemsvalue,0,listofbuyers[i].numitems);
	
	int cnt = 0;
	// set biditemsindex, biditemsvalue for each buyer 
	for (j = rowptr[i]; j < rowptr[i+1]; j++)
	{
	  listofbuyers[i].biditemsindex[cnt] = colptr[j];
	  listofbuyers[i].biditemsvalue[cnt] = valueptr[j];	
	  cnt++;	
	}
#ifdef SORT_BUYERS_ITEM	
	// Sort items in buyer's list in descending order
	SortBuyersItemGreatestToLeast(listofbuyers[i].biditemsindex, 
	                              listofbuyers[i].biditemsvalue,
                                  listofbuyers[i].biditemsindex, 
                                  listofbuyers[i].biditemsvalue,
                                  listofbuyers[i].numitems);
#endif             
  }
		
#ifdef AUCTION_SETUP_TIMING
  t1 = GetTimer()/1000.f;
  printf("Auction Setup Buyers Struct	[sec]:	%f\n",t1);
  StartTimer();
#endif
	
  /* buyer_per_item_cnt is an array of a counters for item_bidder list 
   for each item */
  int *buyer_per_item_cnt;
  _MALLOC(buyer_per_item_cnt,int,matrix_col);
  memset(buyer_per_item_cnt,0,matrix_col);
	
  /* acquire number of bidders for each item through array 
   * "bidder_per_item", and allocate array "listofitems[i].items_bidder"
   * for tracking all buyers for each item. This loop reads array 
   * "bidder_per_item" for each item that is stored in the array 
   * "unique_item" and assigns it to data structure "listofitems" that 
   * will be used for auction algorithm */
  for (i = 0; i < matrix_col; i++)
  {
	listofitems[unique_item[i]].numbidders = bidder_per_item[i];		
  // items_bidder indicates which buyers interested in bidding the item
	_MALLOC(listofitems[unique_item[i]].items_bidder,
			int,
			listofitems[unique_item[i]].numbidders);	
  }
	    
  // assign buyers to item's list in this loop
  for (i = 0; i < matrix_row; i++)
  {	
	// this loop iterates over all items in buyer's row
	for (j = rowptr[i]; j < rowptr[i+1]; j++)
	{
	  // increment index for item's buyer list by one to add new buyer in list
	  int index = buyer_per_item_cnt[colptr[j]]++;
	  listofitems[colptr[j]].items_bidder[index] = i;
	}		
  }
	
#ifdef AUCTION_SETUP_TIMING
  t1 = GetTimer()/1000.f;
  printf("Auction Setup Item Struct  [sec]:  %f\n",t1);
#endif
	
  printf("\n\t\tAuction Parameters Info\n");
  printf("**************************************************************\n");
  printf("delta: %0.6e \t gmin: %0.6e \t gmax: %0.6e\n",delta,gmin,gmax);
  printf("Range:[0,%f]\n",-1*delta*(matrix_row+1));
  printf("**************************************************************\n");
		
#ifdef SHOW_PREAUCTION_INFO
  printf("\nBidders per Item Statistics\n");
  // printing each item with the respective buyers that can bid on that item 	
  for (i = 0; i < matrix_col; i++)
  {
	printf("%d bidders for item %d \t\t", listofitems[i].numbidders, i);
	printf("item %d buyers: ",i);
	for (j = 0; j < listofitems[i].numbidders; j++)
		printf(" %d ", listofitems[i].items_bidder[j]);
		
	printf("\n");
  }

  printf("\n\nBuyer's Item of Interest\n");
  // printing each buyer's nonzero in row including column id
  for (i = 0; i < matrix_row; i++)
  {
	printf("buyer %d: ",i);
	for (j = 0; j < listofbuyers[i].numitems; j++)
	{			
		printf("item %d (%0.4f) ", listofbuyers[i].biditemsindex[j], 
		                           listofbuyers[i].biditemsvalue[j]);
	}
	printf("\n");
  }
#endif	
}
/********************************************************************
						Auction Algorithm
* Description: The algorithm starts of by initializing diagonal, price
* vector and matching vector. The auction proceeds until all buyers are 
* matched. During each iteration, all unmatched buyers are given chance 
* to "bid" on item that will give buyer the most profit. Profit is the
* difference of aij - pj where aij is nonzero value in row of buyer and
* pj is price of item j that buyer wants to bid. Diagonal is updated with
* nonzero value of item j when assigned to buyer.Prior to bidding, each
* buyer searches its item, to find the best and second best column in terms
* of profit. Price vector can be characterized as in increasing positive 
* function during each round of the auction. If item j is transfered from 
* buyer x to buyer y during auction then matching vector M and diagonal are 
* updated along with the buyer's matched/unmatched status. Bid can be 
* calculated as follows --> buyer bid = bid raise + pj + eps + pert
* bid raise = firstprofit - secondprofit
* pj = current price of item j 
* eps = 1/N where N is size of matrix
* pert is price pertubation to avoid price war effect.
**********************************************************************/
void Auction(float diagonalcost)
{	
  int i,j,k;
    
  printf("\n\t Auction Algorithm \n");	
	
  int row, col, value;
  int buyer_assigned_cnt = 0, colid;	

  float pert;	
  int iter_period_print = 100;
  
  int seed;
  seed = time(NULL);
  srand(seed);

  int iter = 0;
  eps = 1 / matrix_row;
  // loop until all buyers are matched
  while (buyer_assigned_cnt < matrix_row)
  {
#ifdef AUCTION_TIMING_INFO
  double t1;
  StartTimer();	
#endif
	iter++;
	
	// loop through all unassigned buyers
	for (i = 0; i < matrix_row; i++)
	{
	  // matched buyers dont bid
	  if (listofbuyers[i].matched == 1)
		  continue;
		
	  int bestcolid,colidvalue;	
	  listofbuyers[i].firstprofit = 0;
	  listofbuyers[i].secondprofit = 0;
	  // find firstprofit of buyer by searching buyer nonzero row
	  for (j = 0; j < listofbuyers[i].numitems; j++)
	  {
	    float aij = listofbuyers[i].biditemsvalue[j];
		int aij_index = listofbuyers[i].biditemsindex[j];
		// firstprofit = max(aij - pj)
		if (aij - pricevector[aij_index] > listofbuyers[i].firstprofit)
		{	
		  listofbuyers[i].firstprofit = aij - pricevector[aij_index];
		  listofbuyers[i].firstprofitindex = aij_index;
		  bestcolid = aij_index;
		  colidvalue = j;
		}
	  }
	  int colidvalue2nd;
	  int bestcolid2nd;
	  // search all items in buyer's row to find second highest profit
	  for (j = 0; j < listofbuyers[i].numitems; j++)
	  {	
		float aij = listofbuyers[i].biditemsvalue[j];
		int aij_index = listofbuyers[i].biditemsindex[j];
	
		// skip best colum
		if (aij_index == bestcolid)
		  continue;				
		
		if (aij - pricevector[aij_index] > listofbuyers[i].secondprofit) 		
		{
		  listofbuyers[i].secondprofit = aij - pricevector[aij_index];
		  listofbuyers[i].secondprofitindex = aij_index;
		  colidvalue2nd = j;
		  bestcolid2nd = aij_index;
		}
	  }
	  
	  float bid;		
      pert=10e-3 *(listofbuyers[i].firstprofit-pricevector[bestcolid]);
	  // bid calculation
	  bid=listofbuyers[i].firstprofit-listofbuyers[i].secondprofit+
	                      pricevector[bestcolid]+eps+pert;		                        
	  // update price of item with new bid
	  pricevector[bestcolid] = bid;
		
	  // search matching vector M for item "j" that buyer i' loses to 
	  // buyer i. Set M[i'] = -1 , diagonalmax[i'] = 0 
	  for (k = 0; k < matrix_row; k++)
	  {
		if (M[k] == bestcolid)
		{  
		  // indicate buyer i' is not matched by -1 in M vector
		  M[k] = -1; 
		  // zero diagonal for i' since it's unmatched 
		  diagonalmax[k] = 0;
		  // indicate buyer i' is unmatched 
		  listofbuyers[k].matched = 0;
		}
	  }	 
	  // update matching vector for buyer i with item j	 
	  M[i] = bestcolid;
	  // update diagonal with nonzero value of item
	  diagonalmax[i] = listofbuyers[i].biditemsvalue[colidvalue];
	  // indicate buyer is matched
	  listofbuyers[i].matched = 1;
	}

	buyer_assigned_cnt = 0;
	// calculate number of buyers assigned
	for (i = 0; i < matrix_row; i++)
	{
	  if (M[i] != -1)
		buyer_assigned_cnt++;
	}
	
#ifdef AUCTION_TIMING_INFO
	if (iter % iter_period_print == 0)
	{
	  t1 = GetTimer()/1000.f;
	  StartTimer();	
	  printf("Iteration %d\t Time: %f\t",iter,t1);
	  printf("buyer assigned: %d/%d\n",buyer_assigned_cnt,matrix_row);
	}
#endif		
  }
  
  diagonalcost = 0;

  // calculating sum of the diagonal
  int matchedcardinalitycnt = 0;
  for (i = 0; i < matrix_row; i++)
  {
	if (M[i] != -1)
	{
	  diagonalcost += diagonalmax[i];
	  matchedcardinalitycnt++;
	}
  }
  printf("Total Iterations: %d\n",iter);
  printf("Matched Cardinality: %d/%d\n",matchedcardinalitycnt,
                                        matrix_row);  
  printf("Total Preprocessed Weights: %e\n", diagonalcost);
}

void ReadMatrix(char *filename)
{	
  int i,j;									
	
  // file pointer used for reading file
  FILE * fp;
	
  fp = fopen(filename,"r");
  if (fp == NULL)
	perror("Error opening file\n");
   
  // reading row, column and nonzeros in matrix
  fscanf(fp,"%d %d %d",&matrix_row,&matrix_col,&matrix_nonzeros);
    
  // allocating 
  _MALLOC(valueptr,float,matrix_nonzeros);
  _MALLOC(origvalue,float,matrix_nonzeros);
  _MALLOC(colptr,int,matrix_nonzeros);	
  _MALLOC(rowptr,int,matrix_row);
  _MALLOC(unique_item,int,matrix_col);
  _MALLOC(bidder_per_item,int,matrix_col);
	
  int x,y;
  int cnt = 0,rowcnt = 0, unique_item_cnt = 0;
  int prev_x = -123456789;
  float value;
	
  gmin = 10e+10;
  gmax = -1; 
	
  // read until end of file
  while (feof(fp) == 0)
  {
	// file format: MatrixRow	MatrixCol	MatrixValue		
	fscanf(fp,"%d %d %e",&x,&y,&value);
	// setting up valueptr in CSR format
	if (fabs(value) < GMIN_THRESHOLD)
	  valueptr[cnt] = GMIN_THRESHOLD;
	else
	  valueptr[cnt] = value;
	
	origvalue[cnt] = value;
	// setting up colptr in CSR format
	colptr[cnt] = y;
	// setting up rowptr in CSR format
	if (prev_x != x)
	  rowptr[rowcnt++] = cnt;
	
	if (gmin > fabs(value) && fabs(value) > GMIN_THRESHOLD)	
	  gmin =fabs(value);
	
	if (gmax < fabs(value))
	  gmax = fabs(value);
		
	// automatically add first unique item 	in unique_item list
	if (unique_item_cnt == 0)
	{
	  unique_item[unique_item_cnt] = y;
	  bidder_per_item[unique_item_cnt]++;
	  unique_item_cnt++;		
	}
	else
	{
	  int unique_flag = 1;
/* loop through all unique items to check if new data entry is unique
	    (i.e different column index) */
	  for (i = 0; i < unique_item_cnt; i++)
	  {
		if (y == unique_item[i])
		{
		  unique_flag = 0;
		  bidder_per_item[i]++;	
		}
	  }
	  
	  // if unique item then add item to end of array	
	  if (unique_flag == 1)
	  {
		unique_item[unique_item_cnt] = y;
		bidder_per_item[unique_item_cnt]++;
		unique_item_cnt++;
	  }
	
	}		
	prev_x = x;	
	cnt++;
  }
	
  rowptr[rowcnt] = matrix_nonzeros;

  for (i = 0; i < unique_item_cnt; i++)
  {
	// last data entry is overcounted by one weird!!!
	if (y == unique_item[i])
	  bidder_per_item[i]--;
  }
	
  delta = log(gmin/gmax);
 _MALLOC(mi,float,matrix_row);

  // compute mi for matrix
  for (i = 0; i < matrix_row; i++)
  {
	float *rowentry;
	int rowentry_cnt = 0;
	_MALLOC(rowentry,float,rowptr[i+1]-rowptr[i]);	
	for (j = rowptr[i]; j < rowptr[i+1]; j++)
	{
	  rowentry[rowentry_cnt++] = fabs(valueptr[j]);
	}
	mi[i] = GetMaxRowValue(rowentry,rowentry_cnt);
	mi[i] = log(mi[i]);
  }

  float f1,f2;

  /* normalize weights between [0,-delta * (N + 1)] where N = # of 
     rows/col in matrix */
  for (i = 0; i < matrix_row; i++)
  {
	for (j = rowptr[i]; j < rowptr[i+1]; j++)
	{
	  f1 = log(fabs(valueptr[j]));
	  f2 = f1 - mi[i] - delta;
		
	  if (valueptr[j] == 0)
		valueptr[j] = -(matrix_row+1)*fabs(log(gmax));
	  else	  
		valueptr[j] = (matrix_row+1)*f2;	  
	}
  }
  
  //close file
  fclose(fp);
}

void PrintList()
{      
  int i,j;
  for (i = 0; i < matrix_row; i++)
  {
	printf("row %d: ",i);	
	for (j = rowptr[i]; j < rowptr[i+1]; j++)
	{
		printf("(%d,%0.2f) ",colptr[j],valueptr[j]);
	}
	printf("\n");
  }
	
  printf("\ncol: ");	
  for (i = 0; i < matrix_nonzeros; i++)
	printf("%d ", colptr[i]);
  
	
  printf("\nrow: ");	
  for (i = 0; i <= matrix_row; i++)
	printf("%d ", rowptr[i]);
  
  printf("\n");
}

void Summary()
{
  int i;
  float sum = 0;	
  
  printf("\n\n\t\t\tAuction Summary\n");
  printf("*********************************************************\n");
  printf("Buyers-Item Mapping:\n");
	
  for (i = 0; i < matrix_row; i++)
    printf("M[%d]: %d\n", i,M[i]);
 
  printf("\n\nDiagonal Array\n");
  for (i = 0; i < matrix_row; i++)
  {
	sum += diagonalmax[i];
	printf("diagonal[%d]: %f\t sum: %e\n",i,diagonalmax[i],sum);
  }
	
}
// Gets maximum value from the array rowlist
float GetMaxRowValue(float *rowlist, int numitems)
{
  float max = -10000000.f;
  int j;
 
  // get max value in row
  for (j = 0; j < numitems; j++)
  {
    if (max < rowlist[j])
      max = rowlist[j];
  }

  return max;
}
// This function sorts buyer's item in descending order
void SortBuyersItemGreatestToLeast(int *sort_item_index, 
              float *sort_item_value,int *item_index, float *item_value, 
              int numitems)
{
  int i,j;
  // loop over all items in buyer's list
  for (i = 0; i < numitems; i++)
  {
	// add first item to array without sorting
	if (i == 0)
	{
	  sort_item_value[i] = item_value[i];
	  sort_item_index[i] = item_index[i];
	}
	else
	{
	  sort_item_value[i] = item_value[i];
	  sort_item_index[i] = item_index[i];
				
	  int temp_int;
	  float temp_float;
      // sort list after adding item by reverse iteration, and swapping 
	  // larger values to the front, and smaller values to the end of list
	  for (j = i; j > 0; j--)
	  {
		if (sort_item_value[j-1] < sort_item_value[j])
		{						
		  temp_float = sort_item_value[j-1];
		  sort_item_value[j-1] = sort_item_value[j];
		  sort_item_value[j] = temp_float;
						
		  temp_int = sort_item_index[j-1];
		  sort_item_index[j-1] = sort_item_index[j];
		  sort_item_index[j] = temp_int;
		}
      }
	}			
  }
}
/*
void shuffle(int *array, size_t n)
{
    int seed;
    seed = time(NULL);
    srand(seed);
    if (n > 1) 
    {
        size_t i;
			for (i = 0; i < n - 1; i++) 
			{
				size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
				int t = array[j];
				array[j] = array[i];
				array[i] = t;
			}
    }
}
*/
