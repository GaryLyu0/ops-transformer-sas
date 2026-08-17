/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "../../../../kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv_metadata.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace aicpu;
using namespace optiling;

namespace {
struct NeedInitCase {
    const char *name;
    const char *layoutQ;
    bool hasCmpKv;
    int64_t cmpTopK;
    std::vector<int32_t> cuSeqlensQ;
    std::vector<int32_t> seqUsedQ;
    int64_t qSeqSize;
    std::vector<int32_t> seqUsedKv;
    uint32_t expected;
};

class KvQuantSparseAttnSharedkvMetadataNeedInitTest :
    public testing::TestWithParam<NeedInitCase> {};

TEST_P(KvQuantSparseAttnSharedkvMetadataNeedInitTest, MatchesLegacyKernelSemantics)
{
    const NeedInitCase &testCase = GetParam();
    const int64_t batchSize = static_cast<int64_t>(testCase.seqUsedKv.size());
    int32_t metadata[SAS_META_SIZE] = {0};

    std::vector<DataType> dataTypes = {
        DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    std::vector<std::vector<int64_t>> shapes = {
        {static_cast<int64_t>(testCase.cuSeqlensQ.size())},
        {0},
        {0},
        {static_cast<int64_t>(testCase.seqUsedQ.size())},
        {batchSize},
        {SAS_META_SIZE}};
    std::vector<void *> data = {
        testCase.cuSeqlensQ.empty() ? nullptr : const_cast<int32_t *>(testCase.cuSeqlensQ.data()),
        nullptr,
        nullptr,
        testCase.seqUsedQ.empty() ? nullptr : const_cast<int32_t *>(testCase.seqUsedQ.data()),
        const_cast<int32_t *>(testCase.seqUsedKv.data()),
        metadata};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "KvQuantSparseAttnSharedkvMetadata", "KvQuantSparseAttnSharedkvMetadata")
        .Input({"act_seq_len_q", dataTypes[0], shapes[0], data[0]})
        .Input({"act_seq_len_ori_kv", dataTypes[1], shapes[1], data[1]})
        .Input({"act_seq_len_cmp_kv", dataTypes[2], shapes[2], data[2]})
        .Input({"seq_used_q", dataTypes[3], shapes[3], data[3]})
        .Input({"seq_used_kv", dataTypes[4], shapes[4], data[4]})
        .Output({"metadata", dataTypes[5], shapes[5], data[5]})
        .Attr("num_heads_q", static_cast<int64_t>(64))
        .Attr("num_heads_kv", static_cast<int64_t>(1))
        .Attr("head_dim", static_cast<int64_t>(128))
        .Attr("batch_size", batchSize)
        .Attr("max_seqlen_q", testCase.qSeqSize)
        .Attr("max_seqlen_kv", static_cast<int64_t>(8))
        .Attr("cmp_topk", testCase.cmpTopK)
        .Attr("cmp_ratio", static_cast<int64_t>(128))
        .Attr("ori_mask_mode", static_cast<int64_t>(4))
        .Attr("ori_win_left", static_cast<int64_t>(127))
        .Attr("layout_q", std::string(testCase.layoutQ))
        .Attr("layout_kv", std::string("PA_ND"))
        .Attr("has_ori_kv", true)
        .Attr("has_cmp_kv", testCase.hasCmpKv);

    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(metadata[GLOBAL_METADATA_BASE + GLOBAL_NEED_INIT_INDEX], testCase.expected);
}

INSTANTIATE_TEST_SUITE_P(
    NeedInitTruthTable,
    KvQuantSparseAttnSharedkvMetadataNeedInitTest,
    testing::Values(
        NeedInitCase{"CfaTndFalse", "TND", true, 0, {0, 1, 2}, {}, 2, {8, 8}, 0},
        NeedInitCase{"CfaTndSeqUsedQConflict", "TND", true, 0, {0, 1, 2}, {9, 9}, 2, {8, 8}, 0},
        NeedInitCase{"CfaTndTrue", "TND", true, 0, {0, 9, 10}, {}, 10, {8, 8}, 1},
        NeedInitCase{"ScfaTndFalse", "TND", true, 512, {0, 1, 2}, {}, 2, {8, 8}, 0},
        NeedInitCase{"ScfaTndTrue", "TND", true, 512, {0, 9, 10}, {}, 10, {8, 8}, 1},
        NeedInitCase{"ScfaTndSeqUsedQConflict", "TND", true, 512, {0, 9, 10}, {1, 1}, 10, {8, 8}, 1},
        NeedInitCase{"SwaTndFalse", "TND", false, 0, {0, 1, 2}, {}, 2, {8, 8}, 0},
        NeedInitCase{"SwaTndTrue", "TND", false, 0, {0, 9, 10}, {}, 10, {8, 8}, 1},
        NeedInitCase{"CfaBsndSeqUsedQConflictFalse", "BSND", true, 0, {}, {9, 9}, 4, {8, 8}, 0},
        NeedInitCase{"CfaBsndSeqUsedQConflictTrue", "BSND", true, 0, {}, {1, 1}, 4, {8, 3}, 1}),
    [](const testing::TestParamInfo<NeedInitCase> &info) { return info.param.name; });
} // namespace
