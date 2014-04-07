/* Name: Shahzeb Siddiqui
*  Email: shahzeb.siddiqui@us.ibm.com
*  Program: Auction Algorithm
*  Description: This program implements the auction algorithm which is 
*  used for solving the assignment problem. In auction, there are 
*  buyers, and items that need to be matched in a manner to provide 
*  maximum global profit for buyers. The input file contains nonzeros 
*  in matrix along with x,y coordinate. The matrix is transformed into a 
*  Compressed Sparse Row format for better memory usage. The algorithm 
*  computes the sum of the diagonal and prints the result after auction.
*/

/* LIBRARIES */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "timer.h"


/* MACRO DEFINITION */
#define AUCTION_SETUP_TIMING
//#define AUCTION_TIMING_INFO
//#define SHOW_PREAUCTION_INFO
//#define AUCTION_SUMMARY


#define _MALLOC(P,T,S) {P=(T*)malloc((S)*sizeof(T));assert(P!= NULL);}
#define _FREE(P) { if ( (P) != NULL) free(P); }
  
/* 				Buyer Struct Details
 * biditemsvalue: list of nonzeros in buyer's row
 * biditemsindex: list of nonzeros index (column) in buyer's row
 * numitems: number of items (# of nonzeros) buyer can bid
 * buyer_rowid: buyer's reference  in terms of row # in matrix
 * matched: matched = 1 buyer is matched to item, otherwise 0
 * matched_itemid: buyer reference to item assigned 
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

	
int nullvalue_i = -100;
float nullvalue_f = -100;

/* matrix parameters */
int matrix_row, matrix_col, matrix_nonzeros;

/* array of buyer structs that representating each row in matrix */
buyer *listofbuyers;

/* array holding diagonal entries in auction to produce maximum sum */
float *diagonalmax;
float *diagonalmax_local;
/* matching matrix for diagonal */
int *M;
int *M_local;
/* Compressed Sparse Row Format arrays */
float *valueptr, *origvalue;
int *colptr;
int *rowptr;
int localsize, start,end, partition;
int evenpartition; 
/* gmin, gmax finds absolute  min and max value in entire matrix,
 * delta = log(gmin/gmax)*/
float gmin, gmax, delta;
float eps;
float *pricevector;
int *itemid_forpricevector;
int *highbidder_forpricevector; 
/* array holding the maximum value of each row which is used for 
 * normalizing weights*/
float *mi;
int *start_boundary, *end_boundary;
double GMIN_THRESHOLD;


/* FUNCTION DEFINITIONS */
void ReadMatrix(char *filename);
void AuctionSetup(float *diagonalcost);
void Auction(float diagonalcost);
void PrintList();
void Summary();
float GetMaxRowValue(float *rowlist, int numitems);


int main(int argc, char ** argv)
{
  int rank, size;
  double t_start,t_end,time, maxtime;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  float diagonalcost;
  GMIN_THRESHOLD = 10e-35;

  if (rank == 0)
  {
	  StartTimer();
	  ReadMatrix(argv[1]);   
	  printf("Time to Pre-processing and Reading  [sec]:  %f\n", 
			 GetTimer()/1000.f);
  }
  MPI_Bcast(&matrix_row,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&matrix_nonzeros,1,MPI_INT,0,MPI_COMM_WORLD);
  StartTimer();
  AuctionSetup(&diagonalcost);
  time = GetTimer()/1000.f;
  MPI_Reduce(&time,&maxtime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  if (rank == 0)
  printf("Time to Setup Auction [sec]:	%f\n", maxtime);

  t_start = MPI_Wtime();
  Auction(diagonalcost);
  t_end = MPI_Wtime();
  time = t_end - t_start;
  MPI_Reduce(&time,&maxtime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  if (rank == 0)
  printf("Time for Auction 	[sec]:	%f\n", maxtime);

  #ifdef AUCTION_SUMMARY
  Summary();
  #endif
  
  MPI_Finalize();
//  PrintList();
  return 0;
}


/********************************************************************
 * Auction Setup
 * 
 * Description: This function initializes and assigns metadata for 
 * buyer and item struct which will be used for Auction algorithm. 
 * Buyers are represented as rows in Matrix and each buyer contains a 
 * list of nonzero entries in its row which represents item that buyer 
 * can bid. Similarly, each item (column in matrix) can be bid by 
 * multiple buyers if there are 2 or more nonzeros in column. The 
 * weights are normalized for numerical stability based on 
 * Anshul/Yinglong's paper.  									
 * *****************************************************************/
void AuctionSetup (float *diagonalcost)
{
  int rank,size;  
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int i,j;  

#ifdef AUCTION_SETUP_TIMING
  double t1,t1max,t2,t2max;
  StartTimer();
#endif

  if (matrix_row % size == 0)
  {
	  	partition = matrix_row / size;
		evenpartition = 1;
  }		
  else
  {
	partition = matrix_row / size + 1;
	evenpartition = 0;
  }
  // setting up start,end index for row partitioning across process
  if (rank == 0)  
	{
		start = 0;
		end = (rank+1)*partition;
	}
  else if (rank == size-1)
	{
		start = rank*partition;	
		end = matrix_row;
	}	
  else
	{
		start = rank*partition;
		end = (rank+1)*partition;
	}
  
  localsize = end - start;
    
  _MALLOC(start_boundary,int,size);
  _MALLOC(end_boundary,int,size);
  
  start_boundary[0] = 0;
  end_boundary[0] = partition;
  
  for (i = 1; i < size; i++)
  {
	  start_boundary[i] = i*partition;
	  if (i != size-1)
	    end_boundary[i] = (i+1)*partition;
	  else
		end_boundary[i] = matrix_row;	
  }
  
  // Initializing listofbuyers and its metadata
  _MALLOC(listofbuyers,buyer,partition);  
  
  // rank 0 stores global M and diagonalmax while all process have 
  // M_local and diagonalmax_local
  
  _MALLOC(M_local,int,partition);
  _MALLOC(diagonalmax_local,float,partition);
  _MALLOC(pricevector,float,partition);
  _MALLOC(itemid_forpricevector,int,partition);
  _MALLOC(highbidder_forpricevector,int,partition);
  
  if (rank != 0)
  {
    _MALLOC(rowptr,int,matrix_row+1);
    _MALLOC(colptr,int,matrix_nonzeros);
    _MALLOC(valueptr,float,matrix_nonzeros);
  }
  MPI_Bcast(rowptr,matrix_row+1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(colptr,matrix_nonzeros,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(valueptr,matrix_nonzeros,MPI_FLOAT,0,MPI_COMM_WORLD);
  for (i = 0; i < partition; i++)
  {
	  diagonalmax_local[i] = 0;
	  M_local[i] = -1;	  
  }
  // initializing pricevector,highbidder_forpricevector,itemid_forpricevector
  for (i = 0; i < partition; i++)
  {
	  pricevector[i] = 0;	  	  	 
	  highbidder_forpricevector[i] = -1;
	  	  
	  if (!evenpartition)
	  {
	    if (rank != size-1)	
	      itemid_forpricevector[i] = start + i;	    
	    // last process has less actual data, thus indicating extra objects with null value
	    else
	    {
		  if (start+i > matrix_row-1)
		    itemid_forpricevector[i] = nullvalue_i;		    
		  else	  
		    itemid_forpricevector[i] = start + i;	    		
	    }	
	  }    
	  else
	    itemid_forpricevector[i] = start + i;	    		
	  
	  //printf("%f \t %d \t %d\n",pricevector[i],highbidder_forpricevector[i],itemid_forpricevector[i]);  	  	  	 
  }

  
  // rank 0 gathering M_local from each process for initialization
  //MPI_Gather(M_local,partition,MPI_INT,M,partition,MPI_INT,0,MPI_COMM_WORLD);  
  //MPI_Gather(diagonalmax_local,partition,MPI_FLOAT,diagonalmax,partition,MPI_FLOAT,0,MPI_COMM_WORLD);  
  
	
  for (i = 0; i < partition; i++)
  {	
	// buyer struct setup
	listofbuyers[i].numitems = 0;
	listofbuyers[i].matched = 0;	
	listofbuyers[i].firstprofit = listofbuyers[i].secondprofit = 0;
	listofbuyers[i].firstprofitindex = listofbuyers[i].secondprofitindex = -1; 	   
	
	if (i+start > matrix_row-1)
	listofbuyers[i].buyer_rowid = nullvalue_i;
	else
	listofbuyers[i].buyer_rowid = i+start;	
  }
  /*
  for (i = 0; i < size; i++)
  {
	  if(rank == i)
	  {
		  for (j = 0; j < partition;j++)
		  printf("rank %d \t buyerid[%d]: %d \t itemid[%d]: %d\n",rank,j,listofbuyers[j].buyer_rowid,j,itemid_forpricevector[j]);
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);*/
  
#ifdef AUCTION_SETUP_TIMING
  t1 = GetTimer()/1000.f;
  MPI_Reduce(&t1,&t1max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  if (rank == 0)
    printf("Auction Setup Allocation/Initialization Time [sec]:	%f\n", t1max);
  StartTimer();
#endif
  
// acquire number of items for each buyer and assign item's list for each buyer
  listofbuyers[end].numitems = rowptr[end+1] - rowptr[end];
  for (i = 0; i < partition; i++)
  {
	  // extra buyer objects due to uneven partition, thus dont need to fill data for object
	  if (listofbuyers[i].buyer_rowid == nullvalue_i)
	    continue;
	    
	  listofbuyers[i].numitems = rowptr[i+1+start] - rowptr[i+start];		  	  
	
	/*if (rank == 0)
	{
		for (i = 0; i < localsize; i++)
		printf("rank %d \t [%d].numitems: %d \t rowptr[%d]: %d \t rowptr[%d]: %d\n",rank,i,listofbuyers[i].numitems,i+1+start,rowptr[i+1+start],i+start,rowptr[i+start]);		    
	}*/	
			
	// biditemsindex store column index for all nonzeros in buyer's row
	_MALLOC(listofbuyers[i].biditemsindex,int,listofbuyers[i].numitems);		  
		
	// biditemsvalue store nonzero value for each buyer's row
	_MALLOC(listofbuyers[i].biditemsvalue,float,listofbuyers[i].numitems);					
	
	memset(listofbuyers[i].biditemsindex,0,listofbuyers[i].numitems);
	memset(listofbuyers[i].biditemsvalue,0,listofbuyers[i].numitems);
	
	int cnt = 0;
	// set biditemsindex, biditemsvalue for each buyer 
	
	for (j = rowptr[i+start]; j < rowptr[i+1+start]; j++)
	{
	  //printf("[%d]:(%d,%f)\n",j,colptr[j],valueptr[j]);	
	  listofbuyers[i].biditemsindex[cnt] = colptr[j];
	  listofbuyers[i].biditemsvalue[cnt] = valueptr[j];	
	  cnt++;	
	}
  }
  
#ifdef AUCTION_SETUP_TIMING
  t2 = GetTimer()/1000.f;
  MPI_Reduce(&t2,&t2max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  if (rank == 0)
    printf("Auction Setup Buyers Struct	[sec]:	%f\n",t2max);
#endif
if (rank == 0)
{
  printf("\n\t\tAuction Parameters Info\n");
  printf("**************************************************************\n");
  printf("delta: %0.6e \t gmin: %0.6e \t gmax: %0.6e\n",delta,gmin,gmax);
  printf("Range:[0,%f]\n",-1*delta*(matrix_row+1));
  printf("**************************************************************\n");
}
#ifdef SHOW_PREAUCTION_INFO
  if (rank == 0)
	printf("\n\nBuyer's Item of Interest\n");
  int procid;
  for (procid = 0; procid < size; procid++)	
  {
	  if (procid == rank)
	  {
	  for (i = 0; i < localsize; i++)
	  {
		printf("buyer %d: ",i+start);
		for (j = 0; j < listofbuyers[i].numitems; j++)
		{			
			printf("item %d (%0.4f) ", listofbuyers[i].biditemsindex[j], 
									   listofbuyers[i].biditemsvalue[j]);
		}
		printf("\n");
	  }
	  }
	MPI_Barrier(MPI_COMM_WORLD);  
  }	  
#endif	
 	
}
/********************************************************************
						Auction Algorithm
* Description: The algorithm starts of by initializing diagonal array
* and setting sum to zero. The auction proceeds until all buyers are 
* matched. During each iteration, all buyers are given chance to "bid" 
* such that the sum of the diagonal is increased. Each buyer searches 
* all items in its list, until the first item produces a diagonal sum 
* greater than current sum. The algorithm guarantees diagonal sum will 
* increase over time until all buyers are matched or no more update 
* occurs according the variable "update". At most two entries in 
* diagonal get updated in the case of matching item with new buyer. 
* Since two buyers are affected then two rows (i.e two indices) in 
* diagonal array are changed. The entry removed from the diagonal array 
* is set to zero.
**********************************************************************/
void Auction(float diagonalcost)
{
  int rank,size;	
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int i,j,k;
    
  if (rank == 0)
  printf("\n\t Auction Algorithm \n");	
	
  int row, col, value;
  int global_buyer_assigned = 0, prev_buyercnt = 0, colid;	
  int iterthreshold = 50; //threshold number of iterations after no improvement in buyercnt
  int itercnt = 0; //this variable will be used for counting until threshold is reached
  float pert;	
  int iter_period_print = 5;
  
  int seed;
  seed = time(NULL);
  srand(seed);

  int iter = 0;
  eps = 1 / matrix_row;
  // loop until all buyers are matched
  int *lost_buyerid;
  _MALLOC(lost_buyerid,int,partition);
  memset(lost_buyerid,0,partition);
  
  int **lost_buyerid_send;
  lost_buyerid_send = (int**)malloc(sizeof(int*)*size);
  //_MALLOC(*lost_buyerid_send,int*,size);
  
  int *lostbuyer_proccnt;
  _MALLOC(lostbuyer_proccnt,int,size);	
  float *exchange_pricevector;
  int *exchange_itemid_pricevector,*exchange_highbidder_pricevector;
  _MALLOC(exchange_pricevector,float,partition);
  _MALLOC(exchange_itemid_pricevector,int,partition);
  _MALLOC(exchange_highbidder_pricevector,int,partition);
  
  while (global_buyer_assigned < matrix_row)
  {
	if (itercnt == iterthreshold)
	  break;  
#ifdef AUCTION_TIMING_INFO
  double tauc, tauc_max;
  StartTimer();	
#endif
	iter++;
	
	int lostbuyercnt = 0;
	
	// loop through all unassigned buyers	
	for (i = 0; i < partition; i++)
	{	
	   
	  // matched buyers dont bid
	  if (listofbuyers[i].matched == 1 || listofbuyers[i].buyer_rowid == nullvalue_i)
	 {
		//printf("rank %d SKIPPED buyer[%d].id= %d\n",rank,i,listofbuyers[i].buyer_rowid); 	  
		continue;
	 }
	 	 	  
	 int startboundary_index,endboundary_index;	  
	 startboundary_index = itemid_forpricevector[0];
	  	 
	  if (evenpartition)
	  {		 
		 endboundary_index = itemid_forpricevector[partition-1];
	  }
	  // last entry is invalid in uneven partition therefore second last element is valid for boundary condition
	  else
	  {
		  if (itemid_forpricevector[partition-1] == nullvalue_i)
		  endboundary_index = itemid_forpricevector[partition-2];
		  else
		  endboundary_index = itemid_forpricevector[partition-1];
	  }	 	  
	 //printf("process %d working on buyer %d\n",rank,listofbuyers[i].buyer_rowid);	
	  int bestcolid = -1,colidvalue;		  
	  int colidvalue2nd = -1, bestcolid2nd;	  
	  listofbuyers[i].firstprofit = 0;
	  listofbuyers[i].secondprofit = 0;
	  // loop through all items in buyer and find best and second best item in terms of profit
	  for (j = 0; j < listofbuyers[i].numitems; j++)
	  {
	    float aij = listofbuyers[i].biditemsvalue[j];
		int aij_index = listofbuyers[i].biditemsindex[j];
		int aij_index_offset = aij_index - startboundary_index;
		if (aij_index < startboundary_index || aij_index > endboundary_index)
		  continue;
		  
		// updating first profit
		if (aij - pricevector[aij_index_offset] > listofbuyers[i].firstprofit)
		{	
		  // skip first iteration for second profit, and assign second profit = first profit before updating first profit 
		  if (listofbuyers[i].firstprofit != 0)
		  {
			  listofbuyers[i].secondprofit = listofbuyers[i].firstprofit;
			  listofbuyers[i].secondprofitindex = listofbuyers[i].firstprofitindex;
			  bestcolid2nd = bestcolid;
			  colidvalue2nd = colidvalue;
		  }
		  listofbuyers[i].firstprofit = aij - pricevector[aij_index_offset];
		  listofbuyers[i].firstprofitindex = aij_index;
		  bestcolid = aij_index;
		  colidvalue = j;
		  
		}
		// update second profit if first profit is not updated but second profit is increased
		else if(aij - pricevector[aij_index_offset] > listofbuyers[i].secondprofit && j > 0)
		{
			listofbuyers[i].secondprofit = aij - pricevector[aij_index_offset];
			bestcolid2nd = aij_index;
			colidvalue2nd = j;
			//listofbuyers[i].secondprofit = listofbuyers[i].firstprofit;
			listofbuyers[i].secondprofitindex = aij_index;
			//bestcolid2nd = bestcolid;
			//colidvalue2nd = colidvalue;
		}
	  }
	  // if local process row cant compute bid by first/second profit 
	  
	 	    
	  
	  //printf("rank %d\t startboundary: %d \t endboundary: %d\n",rank,startboundary_index,endboundary_index);  	  
		// skip bidding if best item not in local process price vector or second best item not found  
		if (bestcolid < startboundary_index || bestcolid > endboundary_index)
		{
		//	printf("rank %d Buyer %d Item Bid %d \t valid range: {%d,%d}\n",rank, listofbuyers[i].buyer_rowid,bestcolid,startboundary_index,endboundary_index);
			continue;
		}		
	  
	  
	  float bid;		
	  int offset = bestcolid - itemid_forpricevector[0];
	  // if only 1 profit found in pricevector, then second profit is half of first profit, to reduce bid increase.   
	  if (listofbuyers[i].secondprofit == 0 && listofbuyers[i].firstprofit != 0)
	  {  
		  listofbuyers[i].secondprofit = listofbuyers[i].firstprofit / 2;
		  
	  }
	     
	  if (listofbuyers[i].firstprofit - listofbuyers[i].secondprofit < 10E-3)
	    pert=10e-1;  
	  
	  if (listofbuyers[i].firstprofit - pricevector[offset] > 0)
	    pert=10e-3 *(listofbuyers[i].firstprofit-pricevector[offset]);   
	  //if (pert < 0)
	  //{
	  //	pert = 0;
	  //}	  	    
	  
      
	  bid=listofbuyers[i].firstprofit-listofbuyers[i].secondprofit+
	                      pricevector[offset]+pert;		                        
	  
//printf("rank %d buyer: %d    Bid Item %d    P1,P2,I1,I2 : (%0.2f,%0.2f,%d,%d)   pert: %f    currbid: %f    newbid: %f   offset: %d\n",rank, listofbuyers[i].buyer_rowid,bestcolid,listofbuyers[i].firstprofit, listofbuyers[i].secondprofit,listofbuyers[i].firstprofitindex,listofbuyers[i].secondprofitindex, pert, pricevector[offset],bid,offset);	  
	  pricevector[offset] = bid;
	  
	  
	  int lostbuyer_local = 0;
	  for (k = 0; k < partition; k++)
	  {
		if (M_local[k] == bestcolid)
		{
			M_local[k] = -1;
			diagonalmax_local[k] = 0;
			listofbuyers[k].matched = 0;
			lostbuyer_local = 1;
			break;
		}			
	  }
	  if (iter == 1)
	  {
	    lostbuyer_local = 1;
	    //highbidder_forpricevector[offset] = i;	    
	  } 
	  // if buyer not local update highbidder vector and lost_buyerid vector
	  if (lostbuyer_local != 1 && highbidder_forpricevector[offset] != -1)
	  {
		  lost_buyerid[lostbuyercnt++] = highbidder_forpricevector[offset];		  		  
		  
	  }
	
	  M_local[i] = bestcolid;
	  diagonalmax_local[i] = listofbuyers[i].biditemsvalue[colidvalue];
	  listofbuyers[i].matched = 1;		  
    
      highbidder_forpricevector[offset] = listofbuyers[i].buyer_rowid;
	  
	  
	  	 
} // end of big for loop
// *************************************************************************************
	MPI_Barrier(MPI_COMM_WORLD);	
	
	
    for (j = 0; j < size; j++)	
		lostbuyer_proccnt[j] = 0;
	  
	
	// get count of all lost buyers on nonlocal process
	if (iter > 1)
	{
		//printf("rank %d lostbuyercnt: %d\t",rank,lostbuyercnt);
		for (i = 0; i < lostbuyercnt; i++)
		{
			for (j = 0; j < size; j++)
			{
				//printf("lostbuyer: %d \t start: %d \t end: %d\n",lost_buyerid[i],start_boundary[j],end_boundary[j]);			
				if (lost_buyerid[i] >= start_boundary[j] && lost_buyerid[i] <= end_boundary[j])
				  lostbuyer_proccnt[j]++;
			}
		}
    }    
	int procid;
	
	procid = 0;
	
	// allocate memory for lost_buyerid_send
	for (j = 0; j < size; j++)
	{
	  if (lostbuyer_proccnt[j] > 0)
	  {
		  //printf("rank %d for proc %d lostcount %d\n",rank, j, lostbuyer_proccnt[j]);
		  _MALLOC(lost_buyerid_send[j],int,lostbuyer_proccnt[j]);
		  
	  }	  
	  
	}
	for (i = 0; i < size; i++)
	{
		if (rank == i)
		  continue;
		int cnt = 0;
		for (j = 0; j < lostbuyercnt; j++)
		{
			//printf("rank %d ------- lostbuyer: %d \t start: %d \t end: %d\n",rank, lost_buyerid[j],start_boundary[i],end_boundary[i]);			
			if (lost_buyerid[j] >= start_boundary[i] && lost_buyerid[j] <= end_boundary[i])
			  lost_buyerid_send[i][cnt++] = lost_buyerid[j];
		}
	}
		
	int lostbuyercnt_recv = -1;
	for (procid = 0; procid < size; procid++)
	{	
		MPI_Scatter(lostbuyer_proccnt,1,MPI_INT,&lostbuyercnt_recv,1,MPI_INT,procid,MPI_COMM_WORLD);	
		
		// current process sending data
		if (rank == procid)
		{					
			// each process send lostbuyer information to all other process
			for (j = 0; j < size; j++)
			{
				if (lostbuyer_proccnt[j] > 0 && procid != j)
				{
					//printf("Proc %d Sending %d Data to Proc %d\t",rank,lostbuyer_proccnt[j],j);
					for (k = 0; k < lostbuyer_proccnt[j]; k++)
					{
						//printf("rank %d sending lost_buyerid_send[%d][%d]: %d\n",rank,j,k,lost_buyerid_send[j][k]);
					}
					MPI_Send(lost_buyerid_send[j],lostbuyer_proccnt[j],MPI_INT,j,0,MPI_COMM_WORLD);
				}
			}
		}
		// other process recieving data
		else
		{
		  int *lostbuyer_recvarray;
		  if (lostbuyercnt_recv > 0)
		  {
			  _MALLOC(lostbuyer_recvarray,int,lostbuyercnt_recv);	
			  //printf("Proc %d Recieving %d Data From Proc %d\t",rank,lostbuyercnt_recv, procid);
		      MPI_Recv(lostbuyer_recvarray,lostbuyercnt_recv, MPI_INT,procid,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		      
		      int k;
		      // update M_local for process with updated lost buyer info
		      for (k = 0; k < lostbuyercnt_recv; k++)
		      {
				//  printf("rank %d lostbuyer_recvarray[%d]: %d\n",rank,k,lostbuyer_recvarray[k]);
				  //if (rank == 1)
				  //printf("rank %d \t start: %d \t end: %d \t before:  M_local[%d]: %d \t  after: M_local[%d]: %d\n",rank,start_boundary[i], lostbuyer_recvarray[k]-start_boundary[0],M_local[lostbuyer_recvarray[k]-start_boundary[0]],lostbuyer_recvarray[k]-start_boundary[0],M_local[lostbuyer_recvarray[k]-start_boundary[0]]);
				  M_local[lostbuyer_recvarray[k]-start_boundary[rank]] = -1; 
				  diagonalmax_local[lostbuyer_recvarray[k]-start_boundary[rank]] = 0;
				  listofbuyers[lostbuyer_recvarray[k]-start_boundary[rank]].matched = 0;
				  
			  }
			  _FREE(lostbuyer_recvarray);
		  }
		}		
		MPI_Barrier(MPI_COMM_WORLD);  	
	}	
	
	for (j = 0; j < size; j++)
	{
	  if (lostbuyer_proccnt[j])
	  {		  
		  _FREE(lost_buyerid_send[j]);
		  
	  }	  	 
	}	
	

	// round robin swapping pricevector and its component data with send/recieve
	if (rank == 0)
	{
		MPI_Send(pricevector,partition,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD);
	    MPI_Send(itemid_forpricevector,partition,MPI_INT,rank+1,0,MPI_COMM_WORLD);
	    MPI_Send(highbidder_forpricevector,partition,MPI_INT,rank+1,0,MPI_COMM_WORLD);
	    
	    MPI_Recv(exchange_pricevector,partition,MPI_FLOAT,size-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	    MPI_Recv(exchange_itemid_pricevector,partition,MPI_INT,size-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	    MPI_Recv(exchange_highbidder_pricevector,partition,MPI_INT,size-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);	
	}
	else if (rank == size -1)
	{
		MPI_Recv(exchange_pricevector,partition,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	    MPI_Recv(exchange_itemid_pricevector,partition,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	    MPI_Recv(exchange_highbidder_pricevector,partition,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	  
		MPI_Send(pricevector,partition,MPI_FLOAT,0,0,MPI_COMM_WORLD);
	    MPI_Send(itemid_forpricevector,partition,MPI_INT,0,0,MPI_COMM_WORLD);
	    MPI_Send(highbidder_forpricevector,partition,MPI_INT,0,0,MPI_COMM_WORLD);
    }
    else
    {
		MPI_Recv(exchange_pricevector,partition,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	    MPI_Recv(exchange_itemid_pricevector,partition,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	    MPI_Recv(exchange_highbidder_pricevector,partition,MPI_INT,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	  
		MPI_Send(pricevector,partition,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD);
	    MPI_Send(itemid_forpricevector,partition,MPI_INT,rank+1,0,MPI_COMM_WORLD);
	    MPI_Send(highbidder_forpricevector,partition,MPI_INT,rank+1,0,MPI_COMM_WORLD);
	}
    //printf("communication complete.\n");
    //MPI_Barrier(MPI_COMM_WORLD);
    
    // copying new data from neighbor process to current pricevector and its components
    memcpy(pricevector,exchange_pricevector,sizeof(float)*partition);
    memcpy(itemid_forpricevector,exchange_itemid_pricevector,sizeof(int)*partition);
    memcpy(highbidder_forpricevector,exchange_highbidder_pricevector,sizeof(int)*partition);
    
    
	    
	int buyer_assigned_cnt = 0;		
	for (i = 0; i < partition; i++)
	{	  	
	  if (M_local[i] != -1)		  
		buyer_assigned_cnt++;
	}
	MPI_Allreduce(&buyer_assigned_cnt,&global_buyer_assigned, 1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);			
	
	if (global_buyer_assigned > prev_buyercnt)
	{
	  prev_buyercnt = global_buyer_assigned;
	  itercnt = 0;
	}  
	else
	{
		itercnt++;
	}  
	
		
		
#ifdef AUCTION_TIMING_INFO
	if (iter % iter_period_print == 0)
	{
	  tauc = GetTimer()/1000.f;
	  MPI_Reduce(&tauc,&tauc_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	  if (rank == 0)
	  {	
	  printf("Iteration %d\t Time: %f\t buyer assigned: %d/%d\n ",iter,tauc_max,global_buyer_assigned,matrix_row);
          }
	  StartTimer();
	 
	}
#endif		
 
  //MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  float diagonalcostlocal = 0;
  float globaldiagonalcost = 0;
	
  // calculating sum of logs for diagonal entries   
  for (i = 0; i < partition; i++)
  {
	if (M_local[i] != -1)
	{
	  diagonalcostlocal += diagonalmax_local[i];		 
	}
  }
  
  MPI_Reduce(&diagonalcostlocal,&globaldiagonalcost,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  if (rank == 0)
  {
	  printf("Total Iterations: %d\n",iter);
	  printf("Matched Cardinality: %d/%d\n",global_buyer_assigned,
											matrix_row);  
	  printf("Total Preprocessed Weights: %e\n", globaldiagonalcost);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
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
  _MALLOC(rowptr,int,matrix_row+1);  
	
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
		
	prev_x = x;	
	cnt++;
  }
	
  rowptr[rowcnt] = matrix_nonzeros;
  // computing delta
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
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int i,procid;
  float localsum = 0,totalsum = 0;	
  
  if (rank == 0)
  {
	  printf("\n\n\t\t\tAuction Summary\n");
	  printf("*********************************************************\n");
	  printf("Buyers-Item Mapping:\n");
  }
  for (procid = 0; procid < size; procid++)
  {	
	  if (rank == procid)
	  {
		  for (i = 0; i < partition; i++)
			printf("M[%d]: %d\n", i,M_local[i]);
	  }	
	  MPI_Barrier(MPI_COMM_WORLD);
  }
  
  if (rank == 0)
	printf("\n\nDiagonal Array\n");
	
  for (procid = 0; procid < size; procid++)
  {	
	  if (rank == procid)
	  {
		  for (i = 0; i < partition; i++)
		  {
			localsum += diagonalmax_local[i];
			printf("diagonal[%d]: %f\n",i+start,diagonalmax_local[i]);
		  }
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
  }	  	  
  MPI_Reduce(&localsum,&totalsum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  if (rank == 0)
    printf("Total Sum: %e\n",totalsum);	
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

