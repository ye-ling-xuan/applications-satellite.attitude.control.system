// Generated from actual frozen V3 outputs. Final independent audit controls publication.
window.INERTIA_DEFENSE_RESULTS = {
  "verified": true,
  "status": "verified",
  "source_core_sha256": "cffa2aea78cd2e9af40645cfc46715388d8acc643000ee195caae780f615ae6a",
  "source_manifest_sha256": "c4138667e3d31004838c3ab2fd6f8744efd87cef4105ffa4f19623b01269782b",
  "source_final_audit_sha256": "0cbc4b3922472cc10048d4640552abbb5b5ed3b774cbf2440371d4e8b78de05d",
  "summary": [
    "本次仅惯量随机化未扩大可用范围。单轴域外成功率由100.0%变为98.9%，共同网格保持11点。",
    "三轴域外成功率由100.0%变为84.4%，共同区间由0.5–1.5缩至0.5–1.1。",
    "强PID域外均100%成功。标称SAC控制代价低54.7%和28.7%，同时稳定更慢。"
  ],
  "boundaries": [
    {
      "axis": "单轴",
      "controller": "PID-N",
      "interval": "0.5–1.5，11个已测点"
    },
    {
      "axis": "单轴",
      "controller": "PID-R",
      "interval": "0.5–1.5，11个已测点"
    },
    {
      "axis": "单轴",
      "controller": "SAC-N共同",
      "interval": "0.5–1.5，11个已测点"
    },
    {
      "axis": "单轴",
      "controller": "SAC-R共同",
      "interval": "0.5–1.5，11个已测点"
    },
    {
      "axis": "三轴",
      "controller": "PID-N",
      "interval": "0.5–1.5，11个已测点"
    },
    {
      "axis": "三轴",
      "controller": "PID-R",
      "interval": "0.5–1.5，11个已测点"
    },
    {
      "axis": "三轴",
      "controller": "SAC-N共同",
      "interval": "0.5–1.5，11个已测点"
    },
    {
      "axis": "三轴",
      "controller": "SAC-R共同",
      "interval": "0.5–1.1，7个已测点"
    }
  ],
  "training": {
    "models": 12,
    "steps": 12000000,
    "episodes": 10920
  },
  "figures": {
    "success": "../../08_惯量域随机化与泛化边界研究/04_图表与结果页面/01_成功率与惯量倍率.png",
    "boundary": "../../08_惯量域随机化与泛化边界研究/04_图表与结果页面/02_经验可用范围.png",
    "tradeoff": "../../08_惯量域随机化与泛化边界研究/04_图表与结果页面/03_精度速度与控制代价.png",
    "transfer": "../../08_惯量域随机化与泛化边界研究/04_图表与结果页面/04_非等比例惯量迁移.png",
    "calibration": "../../08_惯量域随机化与泛化边界研究/04_图表与结果页面/05_校准消融与固定代表轨迹.png"
  },
  "captions": {
    "success": "两轴强PID与校准SAC-N域外均100%。SAC-R单轴98.9%，三轴84.4%；三轴高惯量端68.9%。",
    "boundary": "SAC-N与强PID覆盖至已测0.5–1.5倍网格。三轴SAC-R共同范围缩至0.5–1.1，不代表真实物理边界。",
    "tradeoff": "标称SAC域外控制代价相对PID-N低54.7% / 28.7%，稳定时间分别4.26 / 6.19 s，PID-N为3.26 / 3.23 s。"
  },
  "calibration_note": "SAC主结果含固定零动作校准。三轴原始动作域外成功率均为0%，同检查点校准后为100%与84.4%。",
  "limitations": [
    "三个训练种子、每轴30组初态、离散惯量网格与20 s保持判据下的描述性结果",
    "控制代价是力矩平方积分，较低代价伴随较慢响应，不等于实测省电",
    "固定反馈未输入惯量或在线辨识，真实卫星仍需估计器、执行器和硬件验证"
  ]
};
