#pragma once
#include "simulator.hpp"
#include <string>
#include <vector>

namespace sjtu {

// Concatenate the given matrices vertically (along rows, axis 0) into a single
// matrix stored in shared memory (SRAM). The returned matrix is freshly
// allocated by the allocator and is owned by the caller, who is responsible
// for releasing it once it is no longer needed. All operand matrices must
// already reside in SRAM.
static Matrix *ConcatChain(const std::vector<Matrix *> &parts,
                           GpuSimulator &gpu_sim,
                           MatrixMemoryAllocator &matrix_memory_allocator,
                           const std::string &base_name) {
  Matrix *cur = matrix_memory_allocator.Allocate(base_name);
  gpu_sim.Copy(parts[0], cur, kInSharedMemory);
  for (size_t k = 1; k < parts.size(); ++k) {
    Matrix *nxt = matrix_memory_allocator.Allocate(base_name + "_next");
    gpu_sim.Concat(cur, parts[k], nxt, 0, kInSharedMemory);
    gpu_sim.ReleaseMatrix(cur);
    cur = nxt;
  }
  return cur;
}

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Round i (0-based) works with the first i+1 key/value pairs.
     *   Q : [i+1, d]   (provided by the rater, initially in HBM)
     *   K : each [1, d], V : each [1, d]  (initially in HBM)
     *
     * We compute the attention output over all keys seen so far:
     *   S  = Q * K_all^T               -> [i+1, i+1]
     *   P  = row_softmax(S)            -> [i+1, i+1]
     *   O  = P * V_all                 -> [i+1, d]
     * and commit O (moved back to HBM) as the answer for this round.
     *
     * Each key/value pair is moved to SRAM exactly once, the first time it
     * appears, and kept there for the remaining rounds so we avoid paying the
     * HBM<->SRAM transfer cost repeatedly.
     */

    // Bring this round's fresh query and the newly-visible key/value into
    // shared memory. Previously-seen keys/values are already in SRAM.
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    // Stack K[0..i] and V[0..i] into K_all / V_all, each [i+1, d], in SRAM.
    std::vector<Matrix *> k_parts(keys.begin(), keys.begin() + static_cast<long>(i) + 1);
    std::vector<Matrix *> v_parts(values.begin(),
                                  values.begin() + static_cast<long>(i) + 1);
    Matrix *k_all = ConcatChain(k_parts, gpu_sim, matrix_memory_allocator,
                                "k_all_" + std::to_string(i));
    Matrix *v_all = ConcatChain(v_parts, gpu_sim, matrix_memory_allocator,
                                "v_all_" + std::to_string(i));

    // K_all is [i+1, d]; transpose it in place to [d, i+1] so that
    // Q [i+1, d] * K_all^T [d, i+1] yields the score matrix S [i+1, i+1].
    gpu_sim.Transpose(k_all, kInSharedMemory);

    Matrix *s = matrix_memory_allocator.Allocate("s_" + std::to_string(i));
    gpu_sim.MatMul(current_query, k_all, s);
    // k_all and current_query are no longer needed after S is produced.
    gpu_sim.ReleaseMatrix(k_all);
    gpu_sim.ReleaseMatrix(current_query);

    // For every row r of S, compute the softmax distribution over the i+1
    // keys, then multiply it by V_all to obtain output row r (shape [1, d]).
    // All of these calculations happen in SRAM.
    std::vector<Matrix *> output_rows;
    output_rows.reserve(i + 1);
    for (size_t r = 0; r <= i; ++r) {
      Matrix *s_row = matrix_memory_allocator.Allocate("s_row");
      gpu_sim.GetRow(s, r, s_row, kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.MatExp(s_row, exp_row);
      gpu_sim.ReleaseMatrix(s_row);

      Matrix *denom = matrix_memory_allocator.Allocate("denom");
      gpu_sim.Sum(exp_row, denom);

      Matrix *prob_row = matrix_memory_allocator.Allocate("prob_row");
      gpu_sim.MatDiv(exp_row, denom, prob_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(denom);

      Matrix *out_row = matrix_memory_allocator.Allocate("out_row");
      gpu_sim.MatMul(prob_row, v_all, out_row);
      gpu_sim.ReleaseMatrix(prob_row);

      output_rows.push_back(out_row);
    }
    gpu_sim.ReleaseMatrix(s);
    gpu_sim.ReleaseMatrix(v_all);

    // Stitch the per-row outputs together into the final [i+1, d] answer.
    Matrix *answer =
        ConcatChain(output_rows, gpu_sim, matrix_memory_allocator,
                    "answer_" + std::to_string(i));
    for (Matrix *row : output_rows) {
      gpu_sim.ReleaseMatrix(row);
    }

    // The rater expects the committed answer to reside in GPU HBM. Queue the
    // transfer; it will execute during Run() below, before CommitAnswer.
    gpu_sim.MoveMatrixToGpuHbm(answer);

    /*********************  End of your code *********************/
    gpu_sim.Run(false, &matrix_memory_allocator);
    // Now that all queued instructions have executed, the answer is in HBM.
    rater.CommitAnswer(*answer);
    // CommitAnswer releases the answer matrix automatically.
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
