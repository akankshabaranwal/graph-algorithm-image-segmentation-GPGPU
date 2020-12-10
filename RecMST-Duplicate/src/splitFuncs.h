/*--------------------------------------------------------------------------------------
Copyright (c) 2011 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESSED, IMPLIED OR 
OTHERWISE.

Created by Suryakant Patidar and Parikshit Sakurikar.
--------------------------------------------------------------------------------------*/



#define	MILLION	(30*32*1024)

#include <stdio.h>
#include <math.h>

#include "scanKernel.cu"
#include "scan.cu.h"
#include "splitKernel.cu"

enum	DataSize
{
	BYTE4=0,BYTE8=1,BYTE16=2
};

class	splitSort
{
	private:
		int	NUMELEMENTS; 
		int	NUMBLOCKS;
		
		int	MAX_NUMBINS;
		int	MAX_NUMBITS;
	
		int	NEPB;
		int	SM_USAGE;

		unsigned int	*d_blockHists;
		unsigned int	*d_blockHistsScan;
		unsigned int	*d_blockHistsStore;

	public:
		void	split( unsigned int *, unsigned int *, unsigned int *, unsigned int *, int, int, int );
		void	split8( unsigned int *, unsigned int *, unsigned int *, unsigned int *, int, int, int );
		
		void	split( unsigned long long int *, unsigned long long int *, unsigned long long int *, unsigned long long int *, int, int, int );
		void	split8( unsigned long long int *, unsigned long long int *, unsigned long long int *, unsigned long long int *, int, int, int );
	
		void	setOptimalBlockCount(int);	
		void	initScratchMemory();
		void	freeScratchMemory();
		void	setParams(int, int);
};

void splitSort::setParams(int numElements, int RSIZE)
{
	NUMELEMENTS = numElements;

	MAX_NUMBITS = 8;
	MAX_NUMBINS = (int) pow(2.0,MAX_NUMBITS);

	//Number of Elements handled per Block
	//setOptimalBlockCount(RSIZE);
	//float	batch = (float)NUMELEMENTS/(float)NUMBLOCKS;
	//NEPB = (int)(ceil(batch));
	
	if ( NUMELEMENTS < 3145728 )
		NEPB = 8*1024;
	else
		NEPB = 16*1024;
	float	batch = (float)NUMELEMENTS/(float)NEPB;
	NUMBLOCKS = (int)(ceil(batch));
}

void splitSort::initScratchMemory()
{
	//Histogram
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_blockHists , sizeof(unsigned int) * MAX_NUMBINS * NUMBLOCKS  ));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_blockHistsScan , sizeof(unsigned int) * MAX_NUMBINS * NUMBLOCKS  ));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_blockHistsStore , sizeof(unsigned int) * MAX_NUMBINS * NUMBLOCKS  ));
}

void splitSort::freeScratchMemory()
{	
	CUDA_SAFE_CALL(cudaFree(d_blockHists));
	CUDA_SAFE_CALL(cudaFree(d_blockHistsScan));
	CUDA_SAFE_CALL(cudaFree(d_blockHistsStore));
}

void splitSort::split( unsigned int* d_inputList, unsigned int *d_rankList, unsigned int* d_scratchMem, unsigned int *d_tempList, int numElements, int nKBits, int stBit )
{
	setParams(numElements, BYTE4);
	initScratchMemory();

	int	nPasses = nKBits / MAX_NUMBITS; 
	if ( nPasses * MAX_NUMBITS != nKBits )
		nPasses++;

	printf("NUMELEMENTS:%d\nNUMBLOCKS:%d\nnPasses:%d\nNEPB:%d\n",NUMELEMENTS, NUMBLOCKS, nPasses, NEPB);	
	for ( int i=0; i < nPasses; i++ )
	{
		if ( i < nPasses - 1 )
			split8( d_inputList, d_rankList, d_scratchMem, d_tempList, i, MAX_NUMBITS * i + stBit, (int) pow(2.0, MAX_NUMBITS) - 1 );
		else
			split8( d_inputList, d_rankList, d_scratchMem, d_tempList, i, MAX_NUMBITS * i + stBit, (int) pow(2.0, nKBits - (i * MAX_NUMBITS) ) - 1 );
	}
	
	freeScratchMemory();
}

void splitSort::split8( unsigned int *d_inputList, unsigned int *d_rankList, unsigned int *d_scratchMem, unsigned int *d_tempList, int bStable, int stBit, int bMask )
{	
	SM_USAGE = sizeof(int) * (bMask+1) * 2;
	
	#ifdef	DEBUG
	unsigned int	timer;
	
	float Total=0;
	
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	histogramCalc<<< NUMBLOCKS, 128, SM_USAGE >>>( d_blockHists, d_blockHistsStore, d_inputList, NUMELEMENTS, bMask+1, stBit, NEPB, bMask );
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 1 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	#endif

	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	preallocBlockSums(NUMBLOCKS*(bMask+1));
	prescanArray(d_blockHistsScan, d_blockHists, (bMask+1) * NUMBLOCKS );
	deallocBlockSums();
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 2 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	#endif
	
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	localScatter<<< NUMBLOCKS, (bStable?32:128), SM_USAGE/2 >>>( d_blockHistsStore, d_inputList, d_rankList, d_scratchMem, d_tempList, NUMELEMENTS, bMask+1, stBit, NEPB, bMask, bStable );
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 3 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	#endif

	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	globalScatter<<< NUMBLOCKS, 128, SM_USAGE >>>( d_blockHistsScan, d_blockHistsStore, d_scratchMem, d_tempList, d_inputList, d_rankList, NUMELEMENTS, bMask+1, stBit, NEPB, bMask );
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 4 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	
	printf("%3.3f\n",Total);	

	#endif


}

///// 64 Bit Calls :: Think abt this


void splitSort::split( unsigned long long int* d_inputList, unsigned long long int *d_rankList, unsigned long long int* d_scratchMem, unsigned long long int *d_tempList, int numElements, int nKBits, int stBit )
{
	setParams(numElements, BYTE8);
	initScratchMemory();

	int	nPasses = nKBits / MAX_NUMBITS; 
	if ( nPasses * MAX_NUMBITS != nKBits )
		nPasses++;

	//printf("NUMELEMENTS:%d\nNUMBLOCKS:%d\nnPasses:%d\nNEPB:%d\n",NUMELEMENTS, NUMBLOCKS, nPasses, NEPB);	
	for ( int i=0; i < nPasses; i++ )
	{
		if ( i < nPasses - 1 )
			split8( d_inputList, d_rankList, d_scratchMem, d_tempList, i, MAX_NUMBITS * i + stBit, (int) pow(2.0, MAX_NUMBITS) - 1 );
		else
			split8( d_inputList, d_rankList, d_scratchMem, d_tempList, i, MAX_NUMBITS * i + stBit, (int) pow(2.0, nKBits - (i * MAX_NUMBITS) ) - 1 );
	}
	
	freeScratchMemory();
}

void splitSort::split8( unsigned long long int *d_inputList, unsigned long long int *d_rankList, unsigned long long int *d_scratchMem, unsigned long long int *d_tempList, int bStable, int stBit, int bMask )
{	
	SM_USAGE = sizeof(int) * (bMask+1) * 2;
	
	#ifdef	DEBUG
	unsigned int	timer;

	float Total=0;

	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	histogramCalc<<< NUMBLOCKS, 128, SM_USAGE >>>( d_blockHists, d_blockHistsStore, d_inputList, NUMELEMENTS, bMask+1, stBit, NEPB, bMask );
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 1 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	#endif

	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	preallocBlockSums(NUMBLOCKS*(bMask+1));
	prescanArray(d_blockHistsScan, d_blockHists, (bMask+1) * NUMBLOCKS );
	deallocBlockSums();
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 2 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	#endif
	
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	localScatter<<< NUMBLOCKS, 32, SM_USAGE/2 >>>( d_blockHistsStore, d_inputList, d_rankList, d_scratchMem, d_tempList, NUMELEMENTS, bMask+1, stBit, NEPB, bMask, bStable );
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 3 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	#endif

	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));	
	CUT_SAFE_CALL( cutStartTimer( timer));
	#endif
	globalScatter<<< NUMBLOCKS, 128, SM_USAGE >>>( d_blockHistsScan, d_blockHistsStore, d_scratchMem, d_tempList, d_inputList, d_rankList, NUMELEMENTS, bMask+1, stBit, NEPB, bMask );
	#ifdef	DEBUG
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("Step 4 :: %3.3f\t",cutGetTimerValue(timer));
	Total+=cutGetTimerValue(timer);
	printf("%3.3f\n",Total);	
	#endif
}

void	splitSort::setOptimalBlockCount( int RSIZE )
{
	if ( RSIZE==BYTE8 )
	{
		if ( NUMELEMENTS < 4 * MILLION )
			NUMBLOCKS = 240;
		else if ( NUMELEMENTS < 8 * MILLION )
			NUMBLOCKS = 480;
		else if ( NUMELEMENTS < 16 * MILLION )
			NUMBLOCKS = 960;
		else if ( NUMELEMENTS < 32 * MILLION )
			NUMBLOCKS = 1920;
		else if ( NUMELEMENTS < 80 * MILLION )
			NUMBLOCKS = 3840;
		else
			NUMBLOCKS = 7680;
	}
	else if ( RSIZE==BYTE4 )
	{
		if ( NUMELEMENTS < 7 * MILLION )
			NUMBLOCKS = 240;
		else if ( NUMELEMENTS < 14 * MILLION )
			NUMBLOCKS = 480;
		else if ( NUMELEMENTS < 26 * MILLION )
			NUMBLOCKS = 960;
		else if ( NUMELEMENTS < 64 * MILLION )
			NUMBLOCKS = 1920;
		else if ( NUMELEMENTS < 128 * MILLION )
			NUMBLOCKS = 3840;
		else
			NUMBLOCKS = 7680;
	}
}
