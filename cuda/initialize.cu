#include "stream_manager.h"

#include "utils/fmoe_utils.h"
#include <torch/extension.h>
#include <nccl.h>

#include <c10d/ProcessGroupNCCL.hpp>

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
        // printf("ncclID: %s\n", ncclID.internal);

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        printf("get comm: %d\n", getSize());
        return comm;
    }
};

void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t, int batch, int d_model, int d_hidden) {
    auto smgr = getCudaStreamManager(t.device().index(), batch, d_model, d_hidden); //t.device().index()
    if (smgr->ncclgood) {
        return;
    }
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
    smgr -> group = h;

    smgr->ncclcomm = h->getcomm(t.device());
    // ncclComm_t comm2 = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
//smgr->comms[i] = h->getcomm(t.device());
    }
}

void _ensure_nccl2(c10d::ProcessGroupNCCL& p, c10d::ProcessGroupNCCL& p2, torch::Tensor t, int batch, int d_model, int d_hidden, std::vector<std::vector<int>> topos) {
    auto smgr = getCudaStreamManager(t.device().index(), batch, d_model, d_hidden); //t.device().index()
    if (smgr->ncclgood) {
        return;
    }
    printf("ensure_nccl2");
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
    HackNCCLGroup* h2 = (HackNCCLGroup*)(void*)&p2;
    smgr -> group = h;

    smgr->ncclcomm = h->getcomm(t.device());
    smgr->ncclcomm2 = h2->getcomm(t.device());
    smgr -> topos = topos;
    // ncclComm_t comm2 = h->getcomm(t.device());
    if (smgr->ncclcomm != 0 && smgr->ncclcomm2 != 0 ) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }

    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
//smgr->comms[i] = h->getcomm(t.device());
    }
}