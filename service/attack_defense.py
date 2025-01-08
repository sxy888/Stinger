#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filename: attack_defense.py
# Created: 2024-10-07
# Description: crud for the attack and defense records
import pymysql
import json
from sqlalchemy import create_engine
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from utils import format_duration
from constant.config import DB_CONFIG
pymysql.install_as_MySQLdb()

Base = declarative_base()

class AttackDefenseModel(Base):
    __tablename__ = 'attack_defense'
    id = Column(Integer, primary_key=True)
    dataset = Column(String(64))
    attack_method = Column(String(64))
    defense_method = Column(String(64))
    defense_overhead = Column(Float)
    attack_acc = Column(Float)
    created_time = Column(DateTime)
    finished_time = Column(DateTime)
    defense_cost_time = Column(Integer)
    attack_cost_time = Column(Integer)
    defense_kwargs = Column(JSON)
    use_for_plot = Column(Boolean)

    def to_list(self):
        sdr = ""
        if self.attack_acc is not None and self.attack_acc > 0:
            sdr = str(round(100 - self.attack_acc, 2))
        return [
            self.id,
            self.created_time,
            self.dataset,
            self.defense_method,
            self.defense_overhead,
            self.attack_method,
            sdr,
            format_duration(self.defense_cost_time),
            format_duration(self.attack_cost_time),
            json.dumps(self.defense_kwargs or {}, ensure_ascii=False),
        ]


# 定义数据库连接信息
db_type = 'mysql'  # 数据库类型，例如：postgresql, mysql, sqlite等
username = DB_CONFIG['username']  # 数据库用户名
password = DB_CONFIG['password']  # 数据库密码
host = DB_CONFIG['host']  # 数据库主机地址
port = DB_CONFIG['port']  # 数据库端口
database = DB_CONFIG['database']  # 数据库名

connection_string = f"{db_type}://{username}:{password}@{host}:{port}/{database}"

engine = create_engine(connection_string, echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

class AttackDefenseService:

    @classmethod
    def create_model(cls, data):
        session = Session()
        new_model = AttackDefenseModel(**data)
        session.add(new_model)
        session.commit()
        model_id = new_model.id
        session.close()
        return model_id

    @classmethod
    def read_model(cls, id=None, **kwargs):
        session = Session()
        if id:
            model = session.query(AttackDefenseModel).get(id)
            if model:
                return model
            else:
                raise HTTPException(status_code=404, detail="Model not found")
        else:
            query = session.query(AttackDefenseModel)
            for key, value in kwargs.items():
                query = query.filter(getattr(AttackDefenseModel, key) == value)
            return query.all()
        session.close()

    @classmethod
    def update_model(cls, id, **kwargs):
        session = Session()
        model = session.query(AttackDefenseModel).get(id)
        if model:
            for key, value in kwargs.items():
                setattr(model, key, value)
            session.commit()
            session.close()
        else:
            raise HTTPException(status_code=404, detail="Model not found")

    @classmethod
    def delete_model(cls, id):
        session = Session()
        model = session.query(AttackDefenseModel).get(id)
        if model:
            session.delete(model)
            session.commit()
            session.close()
        else:
            raise HTTPException(status_code=404, detail="Model not found")

    @classmethod
    def query_models(cls, **kwargs):
        session = Session()
        query = session.query(AttackDefenseModel)
        for key, value in kwargs.items():
            if key == "defense_overhead":
                query = query.filter(AttackDefenseModel.defense_overhead < value + 0.01)
                query = query.filter(AttackDefenseModel.defense_overhead > value - 0.01)
            else:
                query = query.filter(getattr(AttackDefenseModel, key) == value)
        data = [r.to_list() for r in query.all()]
        session.close()
        return data


# Dependency

def get_db():
    return AttackDefenseService