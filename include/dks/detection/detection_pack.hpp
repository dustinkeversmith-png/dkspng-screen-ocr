#pragma once
// Pluggable shape / widget classification packs.
//
//   IShapeClassifier   one pack = one classification concern ("widget type", "checkbox state", ...)
//   TemplatePack       exemplar k-NN over ShapeDescriptors; trainable from labelled crops, saved as .dkpk
//   PackRegistry       ordered, mutable set of packs (add / remove / replace at runtime)
//   DetectionStage     pipeline stage: runs every registered pack over the elements it applies to and
//                      attaches {pack.tag_key(), match.tag} SemanticTags
//
// A pack decides what it looks at through applies(): element kinds + size limits, and optionally a
// required tag written by an earlier pack (e.g. a state pack only runs on boxes tagged ui.checkbox).
// The core layout / OCR code never sees any of this.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../pipeline/pipeline_stage.hpp"
#include "shape_descriptor.hpp"

namespace dks::detect {

struct ShapeMatch {
    std::string tag;         // e.g. "ui.checkbox", "checked", "icon.search", "math.integral"
    float confidence = 0.f;  // 0..1
    uint32_t class_id = 0;
};

// What a pack gets to see for one element.
struct Patch {
    GrayView luma;                         // element crop (with a small margin)
    Rect box;                              // element box in frame coordinates
    ElementKind kind = ElementKind::Icon;
    const std::vector<SemanticTag>* tags = nullptr;  // tags written so far (may be null)

    const SemanticTag* tag(const std::string& key) const {
        if (!tags) return nullptr;
        for (const auto& t : *tags)
            if (t.key == key) return &t;
        return nullptr;
    }
};

class IShapeClassifier {
public:
    virtual ~IShapeClassifier() = default;
    virtual std::string name() const = 0;     // unique pack name
    virtual std::string tag_key() const = 0;  // SemanticTag key the pack writes
    virtual bool applies(const Patch& p) const = 0;
    virtual std::vector<ShapeMatch> classify(const Patch& p) const = 0;
};

// Applicability filter shared by the built-in packs.
struct PackScope {
    std::vector<ElementKind> kinds{ElementKind::Icon, ElementKind::Container, ElementKind::Image};
    int min_side = 8, max_side = 160;
    std::string require_key;               // only run when this tag key exists ...
    std::vector<std::string> require_any;  // ... with one of these values (empty = any value)

    bool ok(const Patch& p) const {
        if (std::find(kinds.begin(), kinds.end(), p.kind) == kinds.end()) return false;
        if (std::min(p.box.w, p.box.h) < min_side || std::max(p.box.w, p.box.h) > max_side) return false;
        if (!require_key.empty()) {
            const SemanticTag* t = p.tag(require_key);
            if (!t) return false;
            if (!require_any.empty() && std::find(require_any.begin(), require_any.end(), t->value) == require_any.end())
                return false;
        }
        return true;
    }
};

// ------------------------------------------------------------------------------------ TemplatePack
// Exemplar k-NN. Deterministic: distance ties break by exemplar index, vote ties by class id.
class TemplatePack : public IShapeClassifier {
public:
    TemplatePack(std::string name, std::string tag_key, PackScope scope = {}, int k = 5, ShapeWeights w = {})
        : name_(std::move(name)), key_(std::move(tag_key)), scope_(std::move(scope)), k_(k), w_(w) {}

    std::string name() const override { return name_; }
    std::string tag_key() const override { return key_; }
    PackScope& scope() noexcept { return scope_; }
    const std::vector<std::string>& classes() const noexcept { return classes_; }
    size_t size() const noexcept { return ex_.size(); }

    uint32_t class_id(const std::string& label) {
        for (uint32_t i = 0; i < classes_.size(); ++i)
            if (classes_[i] == label) return i;
        classes_.push_back(label);
        return uint32_t(classes_.size() - 1);
    }
    void add_example(const std::string& label, GrayView crop) { ex_.push_back({describe(crop), class_id(label)}); }
    void add_example(const std::string& label, const ShapeDescriptor& d) { ex_.push_back({d, class_id(label)}); }

    bool applies(const Patch& p) const override { return !ex_.empty() && scope_.ok(p); }

    std::vector<ShapeMatch> classify(const Patch& p) const override { return classify(describe(p.luma)); }

    std::vector<ShapeMatch> classify(const ShapeDescriptor& q) const {
        std::vector<std::pair<float, uint32_t>> best;  // (distance, exemplar)
        best.reserve(size_t(k_) + 1);
        for (uint32_t i = 0; i < ex_.size(); ++i) {
            const float d = distance(q, ex_[i].d, w_);
            if (int(best.size()) == k_ && d >= best.back().first) continue;
            best.emplace_back(d, i);
            std::sort(best.begin(), best.end());
            if (int(best.size()) > k_) best.pop_back();
        }
        // Distance-weighted vote.
        std::vector<float> vote(classes_.size(), 0.f);
        float total = 0.f;
        for (const auto& [d, i] : best) {
            const float wgt = 1.f / (1.f + d);
            vote[ex_[i].cls] += wgt;
            total += wgt;
        }
        std::vector<ShapeMatch> out;
        for (uint32_t c = 0; c < vote.size(); ++c)
            if (vote[c] > 0.f) out.push_back({classes_[c], total > 0 ? vote[c] / total : 0.f, c});
        std::sort(out.begin(), out.end(), [](const ShapeMatch& a, const ShapeMatch& b) {
            return a.confidence != b.confidence ? a.confidence > b.confidence : a.class_id < b.class_id;
        });
        return out;
    }

    // ".dkpk": "DKPK" u32 ver=1 | name | tag_key | u32 k | u32 n_classes {str} | u32 n {u16 cls, f32 aspect, f32 fill, 192 u8}
    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
        auto str = [&](const std::string& s) { u32(uint32_t(s.size())); std::fwrite(s.data(), 1, s.size(), f); };
        std::fwrite("DKPK", 1, 4, f);
        u32(1);
        str(name_);
        str(key_);
        u32(uint32_t(k_));
        u32(uint32_t(classes_.size()));
        for (const auto& c : classes_) str(c);
        u32(uint32_t(ex_.size()));
        for (const auto& e : ex_) {
            const uint16_t c = uint16_t(e.cls);
            std::fwrite(&c, 2, 1, f);
            std::fwrite(&e.d.log_aspect, 4, 1, f);
            std::fwrite(&e.d.fill, 4, 1, f);
            std::fwrite(e.d.hog.data(), 1, kHog, f);
            std::fwrite(e.d.ink.data(), 1, kInk, f);
        }
        return std::fclose(f) == 0;
    }

    static std::shared_ptr<TemplatePack> load(const std::string& path, PackScope scope = {}) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return nullptr;
        bool ok = true;
        auto u32 = [&]() { uint32_t v = 0; ok = ok && std::fread(&v, 4, 1, f) == 1; return v; };
        auto str = [&]() {
            const uint32_t n = u32();
            std::string s(ok ? n : 0, '\0');
            ok = ok && n < (1u << 20) && std::fread(s.data(), 1, n, f) == n;
            return s;
        };
        char magic[4] = {};
        ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DKPK", 4) == 0 && u32() == 1;
        std::shared_ptr<TemplatePack> p;
        if (ok) {
            const std::string name = str(), key = str();
            const int k = int(u32());
            p = std::make_shared<TemplatePack>(name, key, std::move(scope), k);
            const uint32_t nc = u32();
            for (uint32_t i = 0; ok && i < nc; ++i) p->classes_.push_back(str());
            const uint32_t n = u32();
            for (uint32_t i = 0; ok && i < n; ++i) {
                Exemplar e;
                uint16_t c = 0;
                ok = std::fread(&c, 2, 1, f) == 1 && std::fread(&e.d.log_aspect, 4, 1, f) == 1 &&
                     std::fread(&e.d.fill, 4, 1, f) == 1 && std::fread(e.d.hog.data(), 1, kHog, f) == size_t(kHog) &&
                     std::fread(e.d.ink.data(), 1, kInk, f) == size_t(kInk);
                e.cls = c;
                if (ok) p->ex_.push_back(e);
            }
        }
        std::fclose(f);
        return ok ? p : nullptr;
    }

private:
    struct Exemplar {
        ShapeDescriptor d;
        uint32_t cls = 0;
    };
    std::string name_, key_;
    PackScope scope_;
    int k_;
    ShapeWeights w_;
    std::vector<std::string> classes_;
    std::vector<Exemplar> ex_;
};

// ------------------------------------------------------------------------------------ registry + stage
class PackRegistry {
public:
    // Adds or replaces (same name) a pack; packs run in insertion order, so a pack that requires a
    // tag must come after the pack that writes it.
    void add(std::shared_ptr<const IShapeClassifier> pack) {
        remove(pack->name());
        packs_.push_back(std::move(pack));
    }
    bool remove(const std::string& name) {
        const auto n = packs_.size();
        packs_.erase(std::remove_if(packs_.begin(), packs_.end(), [&](const auto& p) { return p->name() == name; }),
                     packs_.end());
        return packs_.size() != n;
    }
    const std::vector<std::shared_ptr<const IShapeClassifier>>& packs() const noexcept { return packs_; }

private:
    std::vector<std::shared_ptr<const IShapeClassifier>> packs_;
};

class DetectionStage : public IPipelineStage {
public:
    explicit DetectionStage(std::shared_ptr<const PackRegistry> registry, int margin = 2, float min_confidence = 0.5f)
        : reg_(std::move(registry)), margin_(margin), min_conf_(min_confidence) {}
    const char* name() const override { return "detection"; }

    void process(AnalysisContext& ctx) const override {
        if (!ctx.has_layout) return;
        const Gray8& luma = ctx.ensure_luma();
        if (ctx.tags.size() < ctx.layout.elements.size()) ctx.tags.resize(ctx.layout.elements.size());
        for (size_t i = 0; i < ctx.layout.elements.size(); ++i) {
            const Element& e = ctx.layout.elements[i];
            Patch p;
            p.box = e.bbox;
            p.kind = e.kind;
            p.luma = luma.cview().sub(e.bbox.inflate(margin_).clip(luma.width(), luma.height()));
            for (const auto& pack : reg_->packs()) {
                p.tags = &ctx.tags[i];  // re-read: earlier packs may have added tags
                if (!pack->applies(p)) continue;
                const auto m = pack->classify(p);
                if (!m.empty() && m.front().confidence >= min_conf_)
                    ctx.tag(i, pack->tag_key(), m.front().tag, m.front().confidence, pack->name());
            }
        }
    }

private:
    std::shared_ptr<const PackRegistry> reg_;
    int margin_;
    float min_conf_;
};

}  // namespace dks::detect
