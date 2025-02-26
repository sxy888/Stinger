#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: run_defender.py
# Created: 2024-10-26
# Description: 运行防御方法

from datetime import datetime
import logging  # noqa
import time
import typing as t # noqa
import click
import os

from constant.enum import DatasetChoice, DefenderChoice, AttackChoice
from defender.factory import DefenderFactory
from loader.base import Config
from utils import set_logging_formatter
from service import AttackDefenseService as db
from defender.stinger.stinger import attack_method_map

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logger = logging.getLogger(__name__)


# 创建一个命令组
@click.group()
def cli():
    pass


def run_defender(config: Config, defense_kwargs: t.Dict[str, t.Any]):
    # 记录程序开始时间
    dry_run =  defense_kwargs.get("dry_run")
    if dry_run:
        logger.warning("空跑防御算法，所生成数据不会保存")
    start = int(time.time())
    defender = DefenderFactory.get_defender(config)
    overhead = defender.run(defense_kwargs)
    if not defense_kwargs.get("dry_run"):
        attack_method = defense_kwargs.get("attack_method")
        if isinstance(attack_method, AttackChoice):
            defense_kwargs["attack_method"] = attack_method.value
        # 生成记录
        defense_kwargs.pop("dry_run", "")
        db.create_model(dict(
            dataset=config.dataset_choice.value,
            defense_method=config.defender_choice.value,
            defense_overhead=overhead,
            defense_cost_time=int(time.time()) - start,
            created_time=datetime.now(),
            defense_kwargs=defense_kwargs,
            use_for_plot=False,
        ))



# Tamaraw Defender
@cli.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--in-anoa-time", "-i", type=click.FLOAT, default=0.012)
@click.option("--out-anoa-time", "-o", type=click.FLOAT, default=0.04)
@click.option("--pad-l", "-l", type=click.INT, default=100)
def tamaraw(dataset, in_anoa_time, out_anoa_time, pad_l):
    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.TAMARAW,
    )
    defense_kwargs = {
        "in_anoa_time": in_anoa_time,
        "out_anoa_time": out_anoa_time,
        "pad_l": pad_l,
        "dry_run": False,
    }
    run_defender(config, defense_kwargs)


@cli.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--ratio", "-r", default=1.0, type=click.FLOAT, help="to scale the hist")
def wtf_pad(dataset, ratio):

    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.WTF_PAD,
    )
    defense_kwargs = {
        "ratio": ratio
    }
    run_defender(config, defense_kwargs)


@cli.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--client-dummy-pkt-num", "-c", type=click.INT, default=600)
@click.option("--server-dummy-pkt-num", "-s", type=click.INT, default=1400)
def front(dataset, client_dummy_pkt_num, server_dummy_pkt_num):

    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.FRONT,
    )
    defense_kwargs = {
        "client_dummy_pkt_num": client_dummy_pkt_num,
        "server_dummy_pkt_num": server_dummy_pkt_num,
    }
    run_defender(config, defense_kwargs)


@cli.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--alpha", "-a", type=click.FLOAT, default=5)
def mockingbird(dataset, alpha):

    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.MOCKING_BIRD,
    )
    defense_kwargs = {
        "alpha": alpha,
    }
    run_defender(config, defense_kwargs)


@cli.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--tau-high", "-h", type=click.FLOAT, default=0.30)
@click.option("--tau-low", "-l", type=click.FLOAT, default=0.05)
@click.option("--overhead", "-o", type=click.FLOAT, default=0.5)
def awa(dataset, tau_high, tau_low, overhead):

    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.AWA,
    )
    defense_kwargs = {
        "tau_high": tau_high,
        "tau_low": tau_low,
        "OH": overhead,
        "dry_run": False
    }
    run_defender(config, defense_kwargs)

@cli.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--tau-high", "-h", type=click.FLOAT, default=0.30)
@click.option("--tau-low", "-l", type=click.FLOAT, default=0.05)
@click.option("--overhead", "-o", type=click.FLOAT, default=0.5)
def awa_same(dataset, tau_high, tau_low, overhead):

    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.AWA,
    )
    defense_kwargs = {
        "tau_high": tau_high,
        "tau_low": tau_low,
        "OH": overhead,
    }
    run_defender(config, defense_kwargs)


@cli.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--attack-method", "-a", default="DF", type=click.Choice(attack_method_map.keys()))
def stinger(dataset, attack_method):

    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.STINGER,
    )
    defense_kwargs = {
        "attack_method": AttackChoice(attack_method),
    }

    if attack_method == "DF":
        config.defender_choice = DefenderChoice.STINGER_DF
    elif attack_method == "AWF-SDAE":
        config.defender_choice = DefenderChoice.STINGER_AWF_SDAE
    elif attack_method == "AWF-CNN":
        config.defender_choice = DefenderChoice.STINGER_AWF_CNN
    elif attack_method == "Var-CNN":
        config.defender_choice = DefenderChoice.STINGER_VAR_CNN
    elif attack_method == "GANDaLF":
        config.defender_choice = DefenderChoice.STINGER_GandLAF

    run_defender(config, defense_kwargs)


@cli.command()
@click.option("--dataset", "-d", default="DF")
@click.option("--oh-min-threshold", "-min", default=0.48, type=click.FLOAT)
@click.option("--oh-max-threshold", "-max", default=0.52, type=click.FLOAT)
def alert(dataset, oh_min_threshold, oh_max_threshold):
    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice.ALERT,
    )
    defense_kwargs = {
        "dry_run": False,
        "oh_min_threshold": oh_min_threshold,
        "oh_max_threshold": oh_max_threshold,
    }
    run_defender(config, defense_kwargs)


if __name__ == "__main__":
    set_loggis   Pr   N)�torchr   �Mod