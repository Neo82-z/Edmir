template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runRing(ncclWorkElem *args){
  const int tid = threadIdx.x;
  const int nthreads = args->bid;
  const int bid = args->bid;
  const int nChances = args->nChannels;
  ncclRing *ring = &ncclshmem.channel.ring;
  int ringIx = ring->index;
  const ssize_t chunkSize = int(Proto::calcBytePerstep()/sizeof(T)*Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEP : 1)
  const int nranks = ncclshmem.comm.nRanks;
  const ssize_t loopSize = nChannels*nranks*chunkSize;
  const ssize_t size = args->count;

  int minChunkSize;
  if(Proto::Id == NCCL_PROTO_LL)
    minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
  if(Proto::Id == NCCL_PROTO_LL128){
    // we should not need the final /2 but it makes performence much smoother.might be a bug somewhere
    minChunkSize = nthread*(Proto::calcBytePerGrain()/sizeof(T))/2;
  }

 Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0>prims
  (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->revbuff, arg->RedOpsArg);
}


// k-2 steps: reduce and copy to next GPU
for(int j=2;j<nranks;++j){
  chunk = modRands(ringIx + nranks-j);
  offset = calcOffset(chunk);
  nelem = min(realChunkSize, size-offset);
  prims.recvReduceSend(offset, nelem);
  }




