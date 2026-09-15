import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata

from src.temporal_dropout import apply_temporal_dropout


class PASTIS_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
        temporal_dropout=None,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        """
        super(PASTIS_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        self.memory = {}
        # 用于存储临时数据
        self.memory_dates = {}
        # 存储与memory相关的日期信息或时间戳
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        # 用于将原始类别标签转换为新的类别标签
        self.target = target
        self.sats = sats
        self.temporal_dropout = temporal_dropout or {}

        # Get metadata
        # 用于读取和处理地理空间元数据以及卫星数据的时间信息
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        # 使用geopandas读取metadata.geojson文件并将其加载为GeoDataFrame。
        # 该文件包含地理空间数据以及每个补丁对应的元信息（如区域标识符、日期信息等）
        # folder:文件所在的目录路径
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        # 将 meta_patch 中的 ID_PATCH 列（假设是唯一的补丁编号）
        # 设置为 GeoDataFrame 的索引，并确保索引的类型为整数（astype(int)）
        self.meta_patch.sort_index(inplace=True)
        # 按索引对 self.meta_patch 进行排序，确保数据按补丁 ID 排列。
        # inplace=True 表示对原始 GeoDataFrame 进行修改，而不返回新的副本。

        self.date_tables = {s: None for s in sats}
        # 用于根据卫星名称填充每颗卫星的日期表
        self.date_range = np.array(range(-200, 600))
        # 创建一个相对于参考日期的天数偏移量
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            # 提取每颗卫星对应的时间序列信息
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            # index 是 self.meta_patch.index，即补丁的 ID。
            # columns 是日期范围 self.date_range（从 -200 到 599 的整数），这些日期可能是与参考日期的相对天数。
            # dtype=int 表示数据类型是整数，用来存储每个日期是否有数据。
            for pid, date_seq in dates.items():
                # 遍历 dates 中的每一项，pid 是补丁ID，date_seq 是该补丁的时间序列。每个补丁有一系列与该卫星相关的时间信息
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                #     检查date_seq是否是字符串类型，使用json.loads将其转化为python对象（字典或列表）
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                # 将 date_seq 转换为 DataFrame。date_seq 是一个字典，将其作为行数据，orient="index" 表示字典的键作为 DataFrame 的索引。
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                # 这一段代码将 date_seq 中的日期字符串（例如，"20210101"）转换为 datetime 对象，并计算它与参考日期 (self.reference_date) 之间的天数差。
                # str(x)[:4] 提取年份，str(x)[4:6] 提取月份，str(x)[6:] 提取日期。
                # 通过 self.reference_date 计算天数差，.days 获取差值的天数。
                date_table.loc[pid, d.values] = 1
            #     在 date_table 中，根据 pid（补丁 ID）和日期（d.values）的天数，将对应位置的值设置为 1，表示该补丁在该日期有数据
            date_table = date_table.fillna(0)
            # 将 date_table 中所有的 NaN 值填充为 0。NaN 可能出现在没有对应日期数据的位置，填充为 0 表示该日期没有数据。
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }
            # 将date_table转换为字典格式，其中每个补丁的ID(index)对应一系列日期的值(d.values)，
            # 然后将这些数据存储到self.date_tables[s]中。s是当前的卫星名称。
            # date_table.to_dict(orient="index")将date_table转换为字典格式，字典的键是补丁
            # ID，值是该补丁的日期数据。np.array(list(d.values()))将日期数据转换为NumPy数组。

        print("Done.")
        # 读取补丁元数据：从 metadata.geojson 文件中加载数据，包含补丁的各种信息（例如 ID 和日期）。
        # 创建日期表：对于每颗卫星，根据每个补丁的日期序列生成一个日期表（date_table），表示该补丁在哪些日期有数据。
        # 处理日期信息：将日期字符串转换为 datetime 对象，并计算与参考日期的天数差。
        # 填充日期表：在日期表中填充数据，表示哪些日期有数据，哪些没有（使用 0 和 1 来标识）。
        # 存储日期表：将处理好的日期表存储在 self.date_tables 字典中，按照卫星名称分开存储。

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        # 筛选折数据：根据提供的 folds 列表筛选 self.meta_patch 中属于指定折的数据，并合并成一个新的 DataFrame。
        # 计算数据集大小：通过 self.meta_patch.shape[0] 获取数据集的补丁数目，并赋值给 self.len。
        # 获取补丁 ID：提取 self.meta_patch 的索引（补丁 ID），并保存到 self.id_patches。

        # Get normalisation values
        if norm:
            self.norm = {}
            # 如果 norm 为 True（表示需要归一化），则初始化一个空字典 self.norm 用于存储归一化参数。
            for s in self.sats:
                with open(
                    os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                #     使用 json.loads() 读取该文件的内容，将其解析为一个 Python 字典（normvals）。
                #     该字典应包含每个折的均值（mean）和标准差（std）
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                # means 是一个列表，存储每个折的均值。
                # stds 是一个列表，存储每个折的标准差。
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                # 将每个折的均值和标准差堆叠成 NumPy 数组，并计算它们的平均值：
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        #         将计算出的均值和标准差转换为 PyTorch 的张量（torch.Tensor）类型，并将它们保存到 self.norm[s] 中
        else:
            self.norm = None
        print("Dataset ready.")

    def __len__(self):
        return self.len
    # 返回数据集中补丁的数量，即 self.len，表示数据集包含多少个样本（补丁）。

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]
    # 该方法返回特定补丁（id_patch）在特定卫星（sat）上的观测日期。具体的操作是通过查询该补丁在该卫星上对应的日期表（date_tables）来实现的

    def __getitem__(self, item):
        id_patch = self.id_patches[item]
        # 根据索引 item 获取当前数据样本对应的补丁 ID (id_patch)。

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}
            # 检查是否启用了缓存机制，如果没有缓存或数据未缓存，则从文件系统加载卫星数据。

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                    / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            # 如果目标是语义分割（semantic），加载并处理对应的目标标签
            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)
            #         从文件加载标签数据（TARGET_{id_patch}.npy）。
            # target[0] 可能是目标的数组，astype(int) 将标签转换为整数。
            # 如果 class_mapping 存在，则对目标标签进行类映射（即将标签转换为目标类别）。

            # 如果目标是实例分割（instance），加载并处理实例相关的注释数据
            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

            # 如果启用了缓存机制（self.cache），将当前加载的数据和目标标签缓存到 self.memory 中，以便后续访问时快速读取。
            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        # 获取特定补丁的日期序列
        # 根据补丁ID获取该补丁在不同卫星上的观测日期，并缓存到 self.memory_dates 中
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        # 如果 mono_date 是一个整数，则直接从数据中选择该日期。
        # 如果 mono_date 是一个日期对象，则计算与 reference_date 的差异，选择最接近该日期的数据。
        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.temporal_dropout:
            data, dates = apply_temporal_dropout(
                data, dates, self.temporal_dropout
            )

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        # 如果只有一个卫星数据，提取该卫星的数据和日期。
        # 返回一个元组，包含数据（data）、日期（dates）以及目标标签（target）。
        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]

        return (data, dates), target

# 从一个日期字典中提取日期信息，将日期转换为以某个 参考日期（reference_date） 为基准的天数差值，并返回这些天数的数组。
def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    # date_dict 是一个字典，其键为索引值，值为日期信息（例如字符串或列表）。
    # from_dict(date_dict, orient="index") 将字典的值视为行数据，以键作为行索引。
    # 最终创建一个以 index 为行索引的 DataFrame。
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    # 将 DataFrame 的第 0 列中的日期转换为 相对于参考日期的天数差。
    # 将处理后的 DataFrame 列（包含天数差）的值作为 NumPy 数组返回
    return d.values

# 计算每个 fold 的数据均值和标准差（用于归一化操作），然后将这些统计值保存为 JSON 文件。
# folder: 数据集文件夹路径，包含所有相关数据。
# sat: 卫星数据的标识符，例如 "Sentinel-1" 或 "Sentinel-2"。
def compute_norm_vals(folder, sat):
    norm_vals = {}
    # 用于存储每个 fold 的均值和标准差，以 fold 为键。
    for fold in range(1, 6):
        # 循环处理每一个 fold 的数据（假定 1 至 5 是 fold 的编号）。
        dt = PASTIS_Dataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        # folder：数据集根目录。
        # norm=False：禁用归一化，因为需要计算原始数据的均值和标准差。
        # folds=[fold]：仅加载当前 fold 的数据。
        # sats=[sat]：仅加载指定卫星的数据。
        means = []
        stds = []
        for i, b in enumerate(dt):
            # 遍历数据集中的每个样本。b是一个样本的数据，为(data, dates), target。
            print("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0][sat]  # T x C x H x W
            # b[0][0][sat]：取出指定卫星的数据。
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            # 将维度从 (T, C, H, W) 转换为 (C, T, H, W)，方便对通道维度操作。
            # contiguous() 确保内存连续性。
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())
        #     data.view(data.shape[0], -1)：
        # 将数据展平为 (C, T * H * W)，其中每行是一个通道的所有像素值。 mean(dim=-1)：计算每个通道的像素均值。
        # std(dim=-1)：计算每个通道的像素标准差。append：将每个样本的均值与标准差保存到对应列表中。

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))
    #     将所有 fold 的统计值写入 JSON 文件

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))
