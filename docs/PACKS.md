# Building your own detection packs

A pack is a set of labelled example crops compiled into a `.dkpk` file. At run time the `DetectionStage` describes
each candidate box with a `ShapeDescriptor` (a colour- and polarity-independent shape vector) and gives it the label of
its nearest examples. Packs need no neural training: adding examples is the training.

This guide goes from screenshots to a working pack. The worked example uses the widget labels in the Zenodo desktop
dataset and reaches 0.92 hold-out accuracy on 8 classes.

## 1. Decide what the pack answers

One pack answers one question, written as one tag key:

| pack | tag key | labels |
|---|---|---|
| widget type | `widget` | `ui.checkbox`, `ui.switch`, `ui.slider`, `icon.search`, ... |
| checkbox state | `widget.state` | `checked`, `unchecked` |
| your own | anything | anything |

Questions that depend on another answer go in a second pack that *requires* the first pack's tag (section 5). Keep the
label sets small and visually distinct; two labels that look the same will be confused no matter how many examples
you add.

## 2. Collect crops

Crops should look like what the pipeline will see: the element box plus about 2 px of surrounding background, at
the real screen scale. Three ways to get them:

**a) Draw boxes in LabelMe** (the tool used for the Zenodo dataset) on your own screenshots, then:

```bash
python tools/make_pack_dataset.py --labelme my_screens/ --out datasets/my_widgets --labels Checkbox,Switch,Slider
```

**b) List boxes in a TSV** (`image <TAB> x <TAB> y <TAB> w <TAB> h <TAB> label`), for example from a script or from
`screen_pipeline --tsv` output you've corrected by hand:

```bash
python tools/make_pack_dataset.py --boxes my_boxes.tsv --out datasets/my_widgets
```

**c) Put image files in folders yourself.** Any PNG, JPEG, BMP, GIF, TIFF or WebP works:

```
datasets/my_widgets/
  ui.checkbox/   a.png  b.png ...
  ui.switch/     ...
  ui.slider/     ...
```

The folder name is the label. Labels may contain dots (`ui.checkbox`).

How many examples: 10 to 30 per class covers most UI themes; rare classes with fewer than 5 examples get low recall.
Include every look you expect (light and dark theme, hover, disabled, high-DPI), because the pack can only match what
it has seen. Crops from the repo's own datasets work too: `--manifest datasets/ui_states` reads the manifest format
(label = column 5 of each crop's `.gt.tsv`).

## 3. Compile and evaluate

```bash
build/train_pack --classes datasets/my_widgets --name my_widgets --tag-key widget --holdout 5
```

* `--holdout 5` first holds out every 5th crop of each class, reports accuracy and the most common confusion per class,
  then trains on *all* crops and writes the pack. Hold-out crops from the same screenshot as training crops make the
  number optimistic; hold out whole screenshots (put them in a separate folder and test with a second pack) for an
  honest estimate.
* `--k 3` is the number of neighbours that vote (3 worked best on Rico; use 1 for very small classes).
* The pack is written to `data/packs/<name>.dkpk` unless `--out` says otherwise.

Worked example on the Zenodo desktop labels:

```bash
python tools/make_pack_dataset.py --labelme datasets/zenodo --out datasets/zenodo_widgets \
    --labels CheckboxChecked,CheckboxUnchecked,Switch,RadioBtnSelected,RadioBtnUnselected,Dropdown,BtnSq,TextInput
build/train_pack --classes datasets/zenodo_widgets --name zenodo_widgets --tag-key widget --holdout 4
```

```
hold-out (every 4-th crop per class): accuracy 0.919 (79 / 86)
  Dropdown                     recall 0.81  (21/26)   most confused with TextInput
  Switch                       recall 0.50  (1/2)     most confused with TextInput
  ...
wrote data/packs/zenodo_widgets.dkpk  (352 exemplars, 8 classes, k=3, tag key 'widget')
```

Reading the confusions: Dropdown vs TextInput are both wide bordered boxes; a few more dropdown examples with the arrow
clearly visible, or a narrower crop around the arrow, would separate them.

## 4. Use the pack

On the command line, every `.dkpk` in the folder is loaded:

```bash
build/screen_pipeline --packs data/packs
```

From C++:

```cpp
auto packs = std::make_shared<dks::detect::PackRegistry>();
packs->add(dks::detect::TemplatePack::load("data/packs/my_widgets.dkpk"));
pipe.add(std::make_shared<dks::detect::DetectionStage>(packs));
// later, per element i:  ctx.find_tag(i, "widget")->value
```

## 5. Scope and chaining

`PackScope` controls which boxes a pack looks at:

```cpp
dks::detect::PackScope scope;
scope.kinds = {dks::ElementKind::Icon, dks::ElementKind::Container};  // layout kinds (default: Icon, Container, Image)
scope.min_side = 8;  scope.max_side = 160;                              // box size in px
scope.require_key = "widget";                                           // only boxes another pack tagged ...
scope.require_any = {"ui.checkbox", "ui.radio"};                       // ... with one of these values
auto state = dks::detect::TemplatePack::load("data/packs/check_state.dkpk", scope);
```

Packs run in the order they were added, so add the pack that writes `widget` before a pack that requires it.
`screen_pipeline --packs` applies this scope automatically to files with `state` in their name.

## 6. Packs in code

The same thing without files:

```cpp
dks::detect::TemplatePack pack("my_widgets", "widget", {}, /*k=*/3);
pack.add_example("ui.checkbox", crop_luma.cview());   // GrayView of a crop
pack.add_example("ui.switch",   other_crop.cview());
pack.save("data/packs/my_widgets.dkpk");
```

For rules that need no examples, implement `dks::detect::IShapeClassifier` (`name`, `tag_key`, `applies`,
`classify`); `dks::detect::CheckStatePack` in `rule_packs.hpp` is a complete example.

## 7. What a pack can and can't do

The descriptor captures outline shape, edge directions, ink placement, aspect ratio and fill. It ignores colour and
light/dark polarity on purpose, so one example covers both themes. It can't read text inside a widget (that's the
text stage's job) and it can't tell apart two widgets that differ only by colour (for example a red vs green status
dot); such a check is a small rule pack.

Measured results for the shipped packs are in `results/packs.txt`: checkbox state 0.915 (trained) and 0.805 (rule),
Rico widget/icon type 0.585 over 23 classes and 0.772 with icons pooled. The Rico pack comes from phone screenshots,
so on desktop screens a pack built from desktop crops (as above) fits much better.
