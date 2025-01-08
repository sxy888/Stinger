#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: run_defender.py
# Created: 2024-10-26
# Description: 运行防御方法

import os
import sys
import time
import click
import logging  # noqa
import typing as t # noqa
from datetime import datetime
from constant.enum import DatasetChoice, DefenderChoice, AttackChoice
from attack.factory import AttackFactory
from defender.factory import DefenderFactory
from loader.base import Config
from utils import set_logging_formatter, get_all_enum_class_values
from service import AttackDefenseService as db

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logger = logging.getLogger(__name__)


defender_choices = click.Choice(
    get_all_enum_class_values(DefenderChoice, append_all=False),
    case_sensitive=False
)
attack_choices = click.Choice(
    get_all_enum_class_values(AttackChoice, append_all=False),
    case_sensitive=False
)

# Run Attack
@click.command()
@click.option("--dataset", "-d", default="DF", help="dataset name")
@click.option("--defender", "-de", type=defender_choices,)
@click.option("--attack", "-a", type=attack_choices, )
@click.option("--ignore", "-i", is_flag=True, help='指定后将不会保存记录到数据库中')
@click.option("--overwrite", "-o", is_flag=True, help='指定后如果存在历史记录将会重新运行攻击并更新记录')
@click.option("--target-overhead", "-to", type=click.FLOAT, default=0.0)
def run_attack(dataset, defender, attack, ignore, overwrite, target_overhead):

    if ignore:
        logger.warning("The attack result won't be saved to db")

    config = Config(
        dataset_choice = DatasetChoice(dataset),
        defender_choice=DefenderChoice(defender),
        attack_choice=AttackChoice(attack),
    )

    overhead_list = [0]
    if config.defender_choice != DefenderChoice.NO_DEF:
        overhead_list = DefenderFactory.get_overhead_list(config)
        if not overhead_list:
            logger.error(f"Defender {defender} has no overhead list")
            logger.error("Please run `python run_defender.py` to generate data")
            sys.exit(1)
    # record the attack record
    for overhead in overhead_list:
        # print(target_overhead, overhead)
        if target_overhead != 0.0:
            if target_overhead != overhead:
                continue
        config.defense_overhead = overhead
        # check the weather has the record in the same config
        filters = dict(
            dataset=config.dataset_choice.value,
            defense_method=config.defender_choice.value,
            defense_overhead=overhead,
            attack_method=config.attack_choice.value,
            use_for_plot=1,
        )
        # 保证一个 overhead 下一个攻击方法只跑 1 次
        old_records = db.query_models(**filters)
        if len(old_records) > 0:
            if overwrite is False:
                logger.info(f"the overhead {overhead} has already been executed, skip!")
                continue
            else:
                for record in old_records:
                    db.delete_model(record[0])
                old_records = []
        start = int(time.time())
        acc = AttackFactory.get_attack(config).run()
        acc = round(acc * 100, 2)
        if ignore is False:
            db.create_model(dict(
                **filters,
                created_time=datetime.now(),
                attack_acc=acc,
                attack_cost_time=int(time.time()) - start,
            ))
        else:
            logger.info("ignore 模式, 结果不会保存到数据库中")

if __name__ == "__main__":
    set_logging_formatter("DEBUG")
    run_attack()


