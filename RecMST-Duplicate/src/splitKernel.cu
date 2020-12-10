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


#define	WARPSIZE 	32

__global__ void histogramCalc ( unsigned int *blockHists, unsigned int *blockHistStore,  unsigned int *inputList, int NUMELEMENTS, int NUMBINS, int NUMBITS, int NEPB, int BITMASK )
{
	int	blockStartIndex = blockIdx.x * NEPB;
	int	blockEndIndex = blockStartIndex + NEPB;
	blockStartIndex += threadIdx.x;
	if ( blockEndIndex > NUMELEMENTS )
		blockEndIndex = NUMELEMENTS;

	extern __shared__ int 	sharedmem[];
	unsigned int* 		s_Hist = (unsigned int *)&sharedmem;
	
	unsigned int	data, data4;

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		s_Hist[pos] = 0;

	__syncthreads();

	for (int pos = blockStartIndex; pos < blockEndIndex; pos+=blockDim.x ) 
	{
		data4 = inputList[pos];
		data = ( data4 >> NUMBITS ) & BITMASK;
		atomicInc(&s_Hist[data],999999999);
	}

	__syncthreads();

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		blockHists[ blockIdx.x + gridDim.x * pos ] = s_Hist[pos];

	int	val;
	//Prefix Sum
	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
	{
		val = 0;
		for ( int j=0; j <= pos-1; j++ )
			val += s_Hist[j];
		s_Hist[NUMBINS+pos] = val;
	}
	
	__syncthreads();

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		blockHistStore[ blockIdx.x * NUMBINS + pos ] = s_Hist[NUMBINS+pos];
}

__global__ void localScatter( unsigned int *blockHistStore, unsigned int *inputList, unsigned int *rankList, unsigned int *elementList, unsigned int *tempList, int NUMELEMENTS, int NUMBINS, int NUMBITS, int NEPB, int BITMASK, int tsPass )
{
	int	blockStartIndex = blockIdx.x * NEPB;
	int	blockEndIndex = blockStartIndex + NEPB;
	blockStartIndex += threadIdx.x;
	if ( blockEndIndex > NUMELEMENTS )
		blockEndIndex = NUMELEMENTS;

	extern __shared__       int sharedmem[];
	unsigned int* s_Hist =      (unsigned int *)&sharedmem;

	unsigned int	index = 0, data4, data, val;

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		s_Hist[pos] = blockHistStore[blockIdx.x * NUMBINS + pos];

	__syncthreads();

	int 	gIndex = blockIdx.x * NEPB;
	//Scatter Sorted array
	for (int pos = blockStartIndex; pos < blockEndIndex; pos+=blockDim.x ) 
	{
		data4 = inputList[pos];
		val = rankList[pos];
		data = ( data4 >> NUMBITS ) & BITMASK;
		
		//if ( !tsPass )	
			index = atomicInc(&s_Hist[data], 9999999);
		/*else
		{
			for ( int i=0; i < WARPSIZE; i++ )
			{
				if ( threadIdx.x == i )
				{
					index = s_Hist[data];
					s_Hist[data]++;
				}
			}	
		}*/

		elementList[gIndex + index] = data4;
		tempList[gIndex + index] = val;
	}

}

__global__  void globalScatter( unsigned int *blockHistScan, unsigned int *blockHistStore, unsigned int *elementList, unsigned int *tempList, unsigned int *sortedArray, unsigned int *rankList, int NUMELEMENTS, int NUMBINS, int NUMBITS, int NEPB, int BITMASK )
{
	int	blockStartIndex = blockIdx.x * NEPB;
	int	blockEndIndex = blockStartIndex + NEPB;
	blockStartIndex += threadIdx.x;
	if ( blockEndIndex > NUMELEMENTS )
		blockEndIndex = NUMELEMENTS;

	extern __shared__       int sharedmem[];
	unsigned int* s_Hist =      (unsigned int *)&sharedmem;

	unsigned int	data4, data, val;

	//Global Scan Values Load
	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
	{
		s_Hist[pos] = blockHistScan[ blockIdx.x + gridDim.x * pos ];
		s_Hist[NUMBINS+pos] = blockHistStore[ blockIdx.x * NUMBINS + pos ];
	}

	__syncthreads();

	for (int pos = blockStartIndex; pos < blockEndIndex; pos+=blockDim.x ) 
	{
		data4 = elementList[pos];
		val = tempList[pos];
		data = ( data4 >> NUMBITS ) & BITMASK;
		sortedArray[s_Hist[data] + (pos - (blockIdx.x * NEPB) ) - s_Hist[NUMBINS+data]] = data4;
		rankList[s_Hist[data] + (pos - (blockIdx.x * NEPB) ) - s_Hist[NUMBINS+data]] = val;
	}
}

//64 bit Kernels 

__global__ void histogramCalc ( unsigned int *blockHists, unsigned int *blockHistStore,  unsigned long long int *inputList, int NUMELEMENTS, int NUMBINS, int NUMBITS, int NEPB, int BITMASK )
{
	int	blockStartIndex = blockIdx.x * NEPB;
	int	blockEndIndex = blockStartIndex + NEPB;
	blockStartIndex += threadIdx.x;
	if ( blockEndIndex > NUMELEMENTS )
		blockEndIndex = NUMELEMENTS;

	extern __shared__ int 	sharedmem[];
	unsigned int* 		s_Hist = (unsigned int *)&sharedmem;
	
	unsigned long long int	data, data4;

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		s_Hist[pos] = 0;

	__syncthreads();

	for (int pos = blockStartIndex; pos < blockEndIndex; pos+=blockDim.x ) 
	{
		data4 = inputList[pos];
		data = ( data4 >> NUMBITS ) & BITMASK;
		atomicInc(&s_Hist[data],999999999);
	}

	__syncthreads();

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		blockHists[ blockIdx.x + gridDim.x * pos ] = s_Hist[pos];

	int	val;
	//Prefix Sum
	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
	{
		val = 0;
		for ( int j=0; j <= pos-1; j++ )
			val += s_Hist[j];
		s_Hist[NUMBINS+pos] = val;
	}
	
	__syncthreads();

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		blockHistStore[ blockIdx.x * NUMBINS + pos ] = s_Hist[NUMBINS+pos];
}

__global__ void localScatter( unsigned int *blockHistStore, unsigned long long int *inputList, unsigned long long int *rankList, unsigned long long int *elementList, unsigned long long int *tempList, int NUMELEMENTS, int NUMBINS, int NUMBITS, int NEPB, int BITMASK, int tsPass )
{
	int	blockStartIndex = blockIdx.x * NEPB;
	int	blockEndIndex = blockStartIndex + NEPB;
	blockStartIndex += threadIdx.x;
	if ( blockEndIndex > NUMELEMENTS )
		blockEndIndex = NUMELEMENTS;

	extern __shared__       int sharedmem[];
	unsigned int* s_Hist =      (unsigned int *)&sharedmem;

	unsigned int	index = 0;
	unsigned long long int	data4, data, val;

	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
		s_Hist[pos] = blockHistStore[blockIdx.x * NUMBINS + pos];

	__syncthreads();

	int 	gIndex = blockIdx.x * NEPB;
	//Scatter Sorted array
	for (int pos = blockStartIndex; pos < blockEndIndex; pos+=blockDim.x ) 
	{
		data4 = inputList[pos];
		val = rankList[pos];
		data = ( data4 >> NUMBITS ) & BITMASK;
	
		//if ( !tsPass )	
			index = atomicInc(&s_Hist[data], 9999999);
		/*else
		{
			for ( int i=0; i < 32; i++ )
			{
				if ( threadIdx.x == i )
				{
					index = s_Hist[data];
					s_Hist[data]++;
				}
			}	
		}*/
				
		elementList[gIndex + index] = data4;
		tempList[gIndex + index] = val;
	}

}

__global__  void globalScatter( unsigned int *blockHistScan, unsigned int *blockHistStore, unsigned long long int *elementList, unsigned long long int *tempList, unsigned long long int *sortedArray, unsigned long long int *rankList, int NUMELEMENTS, int NUMBINS, int NUMBITS, int NEPB, int BITMASK )
{
	int	blockStartIndex = blockIdx.x * NEPB;
	int	blockEndIndex = blockStartIndex + NEPB;
	blockStartIndex += threadIdx.x;
	if ( blockEndIndex > NUMELEMENTS )
		blockEndIndex = NUMELEMENTS;

	extern __shared__       int sharedmem[];
	unsigned int* s_Hist =      (unsigned int *)&sharedmem;

	unsigned long long int	data4, data, val;

	//Global Scan Values Load
	for(int pos = threadIdx.x; pos < NUMBINS; pos += blockDim.x)
	{
		s_Hist[pos] = blockHistScan[ blockIdx.x + gridDim.x * pos ];
		s_Hist[NUMBINS+pos] = blockHistStore[ blockIdx.x * NUMBINS + pos ];
	}

	__syncthreads();

	for (int pos = blockStartIndex; pos < blockEndIndex; pos+=blockDim.x ) 
	{
		data4 = elementList[pos];
		val = tempList[pos];
		data = ( data4 >> NUMBITS ) & BITMASK;
		sortedArray[s_Hist[data] + (pos - (blockIdx.x * NEPB) ) - s_Hist[NUMBINS+data]] = data4;
		rankList[s_Hist[data] + (pos - (blockIdx.x * NEPB) ) - s_Hist[NUMBINS+data]] = val;
	}
}

