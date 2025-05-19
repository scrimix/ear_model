#pragma once
#include <vector>
#include <unordered_set>
#include <iostream>

struct AccuracyStats {
    int TP = 0;
    int FP = 0;
    int FN = 0;

    void update(const std::vector<int>& ground, const std::vector<int>& pred) {
        std::unordered_set<int> gt(ground.begin(), ground.end());
        std::unordered_set<int> pr(pred.begin(), pred.end());

        for (int p : pr) {
            if (gt.count(p)) TP++;
            else FP++;
        }

        for (int g : gt) {
            if (!pr.count(g)) FN++;
        }
    }

    double precision() const {
        return (TP + FP) ? double(TP) / (TP + FP) : 0.0;
    }

    double recall() const {
        return (TP + FN) ? double(TP) / (TP + FN) : 0.0;
    }

    double f1() const {
        double p = precision();
        double r = recall();
        return (p + r) ? 2 * p * r / (p + r) : 0.0;
    }

    void print() const {
        std::cout << "TP: " << TP << ", FP: " << FP << ", FN: " << FN << "\n";
        std::cout << "Precision: " << precision()
                  << ", Recall: " << recall()
                  << ", F1 Score: " << f1() << "\n";
    }
};
