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


#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

	inline bool 
isPowerOfTwo(int n)
{
	return ((n&(n-1))==0) ;
}

	inline int 
floorPow2(int n)
{
#ifdef WIN32
	// method 2
	return 1 << (int)logb((float)n);
#else
	// method 1
	// float nf = (float)n;
	// return 1 << (((*(int*)&nf) >> 23) - 127); 
	int exp;
	frexp((float)n, &exp);
	return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256

unsigned int** g_scanBlockSums;
unsigned int g_numEltsAllocated = 0;
unsigned int g_numLevelsAllocated = 0;

void preallocBlockSums(unsigned int maxNumElements)
{
	assert(g_numEltsAllocated == 0); // shouldn't be called 

	g_numEltsAllocated = maxNumElements;

	unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
	unsigned int numElts = maxNumElements;

	int level = 0;

	do
	{       
		unsigned int numBlocks = 
			max(1, (int)ceil((float)numElts / (2.f * blockSize)));
		if (numBlocks > 1)
		{
			level++;
		}
		numElts = numBlocks;
	} while (numElts > 1);

	g_scanBlockSums = (unsigned int**) malloc(level * sizeof(unsigned int*));
	g_numLevelsAllocated = level;

	numElts = maxNumElements;
	level = 0;

	do
	{       
		unsigned int numBlocks = 
			max(1, (int)ceil((float)numElts / (2.f * blockSize)));
		if (numBlocks > 1) 
		{
			cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(unsigned int));
		}
		numElts = numBlocks;
	} while (numElts > 1);

	/*CUT_CHECK_ERROR("preallocBlockSums");*/
}

void deallocBlockSums()
{
	for (int i = 0; i < g_numLevelsAllocated; i++)
	{
		cudaFree(g_scanBlockSums[i]);
	}

	/*CUT_CHECK_ERROR("deallocBlockSums");*/

	free((void**)g_scanBlockSums);

	g_scanBlockSums = 0;
	g_numEltsAllocated = 0;
	g_numLevelsAllocated = 0;
}


void prescanArrayRecursive(unsigned int *outArray, 
		unsigned int *inArray, 
		int numElements, 
		int level)
{
	unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
	unsigned int numBlocks = 
		max(1, (int)ceil((float)numElements / (2.f * blockSize)));
	unsigned int numThreads;

	if (numBlocks > 1)
		numThreads = blockSize;
	else if (isPowerOfTwo(numElements))
		numThreads = numElements / 2;
	else
		numThreads = floorPow2(numElements);

	unsigned int numEltsPerBlock = numThreads * 2;

	// if this is a non-power-of-2 array, the last block will be non-full
	// compute the smallest power of 2 able to compute its scan.
	unsigned int numEltsLastBlock = 
		numElements - (numBlocks-1) * numEltsPerBlock;
	unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
	unsigned int np2LastBlock = 0;
	unsigned int sharedMemLastBlock = 0;

	if (numEltsLastBlock != numEltsPerBlock)
	{
		np2LastBlock = 1;

		if(!isPowerOfTwo(numEltsLastBlock))
			numThreadsLastBlock = floorPow2(numEltsLastBlock);    

		unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
		sharedMemLastBlock = 
			sizeof(unsigned int) * (2 * numThreadsLastBlock + extraSpace);
	}

	// padding space is used to avoid shared memory bank conflicts
	unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
	unsigned int sharedMemSize = 
		sizeof(unsigned int) * (numEltsPerBlock + extraSpace);

/*#ifdef DEBUG
	if (numBlocks > 1)
	{
		assert(g_numEltsAllocated >= numElements);
	}
#endif*/

	// setup execution parameters
	// if NP2, we process the last block separately
	dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
	dim3  threads(numThreads, 1, 1);

	// make sure there are no CUDA errors before we start
	/*CUT_CHECK_ERROR("prescanArrayRecursive before kernels");*/

	// execute the scan
	if (numBlocks > 1)
	{
		prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
				inArray, 
				g_scanBlockSums[level],
				numThreads * 2, 0, 0);
		/*CUT_CHECK_ERROR("prescanWithBlockSums");*/
		if (np2LastBlock)
		{
			prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
				(outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
				 numBlocks - 1, numElements - numEltsLastBlock);
			/*CUT_CHECK_ERROR("prescanNP2WithBlockSums");*/
		}

		// After scanning all the sub-blocks, we are mostly done.  But now we 
		// need to take all of the last values of the sub-blocks and scan those.  
		// This will give us a new value that must be sdded to each block to 
		// get the final results.
		// recursive (CPU) call
		prescanArrayRecursive(g_scanBlockSums[level], 
				g_scanBlockSums[level], 
				numBlocks, 
				level+1);

		uniformAdd<<< grid, threads >>>(outArray, 
				g_scanBlockSums[level], 
				numElements - numEltsLastBlock, 
				0, 0);
		/*CUT_CHECK_ERROR("uniformAdd");*/
		if (np2LastBlock)
		{
			uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, 
					g_scanBlockSums[level], 
					numEltsLastBlock, 
					numBlocks - 1, 
					numElements - numEltsLastBlock);
			/*CUT_CHECK_ERROR("uniformAdd");*/
		}
	}
	else if (isPowerOfTwo(numElements))
	{
		prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
				0, numThreads * 2, 0, 0);
		/*CUT_CHECK_ERROR("prescan");*/
	}
	else
	{
		prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
				0, numElements, 0, 0);
		/*CUT_CHECK_ERROR("prescanNP2");*/
	}
}

void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
	prescanArrayRecursive(outArray, inArray, numElements, 0);
}


#endif // _PRESCAN_CU_
