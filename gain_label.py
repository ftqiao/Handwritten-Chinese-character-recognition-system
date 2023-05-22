import pickle
# 原始数据集 1028个汉字
# 最终可用1022个
my_lst = ['烈', '业', '士', '围', '席', '权', '曲', '错', '着', '联', '磨', '米', '较', '并', '得', '好', '巴', '芽', '神', '范', '可', '艰', '成', '示', '脑', '粗', '验', '奋',
          '它', '建', '汽', '按', '替', '石', '路', '配', '斜', '已', '弱', '异', '状', '盐', '涮', '战', '积', '记', '辟', '半', '指', '兵', '练', '宽', '尔', '果', '回', '穗',
          '冲', '玉', '适', '局', '热', '七', '磁', '刻', '株', '阶', '困', '演', '落', '但', '壤', '副', '儒', '投', '苦', '翻', '中', '讲', '种', '最', '碳', '顺', '啊', '企',
          '坏', '利', '没', '级', '呢', '序', '砂', '针', '承', '是', '构', '型', '吗', '负', '具', '据', '脸', '志', '那', '事', '性', '星', '制', '妈', '两', '角', '滚', '体',
          '时', '之', '促', '间', '控', '另', '量', '极', '介', '呆', '术', '护', '位', '某', '引', '宣', '德', '差', '告', '尽', '火', '机', '卵', '商', '第', '听', '甲', '井',
          '先', '缩', '儿', '唯', '甚', '弹', '致', '脚', '寸', '视', '苗', '兰', '脱', '举', '食', '什', '混', '历', '末', '损', '证', '加', '玩', '思', '把', '称', '莅', '从',
          '责', '寨', '钻', '革', '黑', '征', '普', '李', '触', '还', '欢', '核', '润', '官', '诸', '刚', '大', '限', '列', '注', '州', '左', '朝', '映', '件', '使', '呀', '长',
          '话', '周', '千', '红', '彻', '卫', '字', '湿', '几', '贺', '况', '完', '简', '家', '申', '白', '降', '首', '短', '们', '侃', '新', '皮', '终', '留', '纪', '及', '紧',
          '晚', '侯', '任', '光', '荣', '冷', '猪', '门', '偏', '木', '田', '伍', '内', '伟', '环', '带', '充', '措', '黄', '践', '兮', '处', '页', '方', '花', '分', '略', '评',
          '立', '修', '威', '粉', '省', '际', '未', '须', '十', '亚', '既', '且', '宫', '编', '期', '汉', '含', '眼', '初', '均', '稳', '议', '泵', '司', '正', '夏', '宋', '陈',
          '包', '侧', '难', '往', '望', '刺', '盖', '律', '敌', '露', '停', '换', '益', '总', '请', '吃', '沙', '暴', '率', '贯', '元', '贫', '沿', '类', '起', '础', '干', '划',
          '待', '微', '防', '秋', '养', '彪', '只', '逐', '百', '实', '树', '了', '亩', '插', '互', '巩', '京', '六', '度', '束', '版', '油', '绝', '论', '会', '维', '帮', '稻',
          '城', '考', '能', '纹', '增', '值', '免', '意', '服', '棉', '焊', '夜', '血', '单', '高', '严', '响', '倍', '器', '液', '态', '谈', '吸', '化', '瓦', '纲', '国', '活',
          '虽', '公', '则', '夫', '物', '府', '布', '她', '产', '箱', '劳', '晶', '陕', '杀', '剥', '段', '和', '铁', '通', '基', '余', '工', '所', '细', '久', '博', '胞', '其',
          '写', '王', '信', '料', '罗', '空', '卡', '员', '上', '始', '马', '专', '厚', '各', '断', '里', '氧', '图', '仍', '补', '哪', '海', '让', '他', '同', '岩', '便', '圈',
          '定', '随', '在', '客', '题', '自', '破', '确', '斯', '子', '牙', '盛', '管', '消', '院', '农', '即', '走', '效', '跟', '认', '西', '言', '埃', '宝', '就', '或', '析',
          '美', '品', '调', '过', '北', '继', '价', '磷', '训', '靠', '质', '试', '丝', '住', '协', '楚', '层', '送', '又', '爱', '提', '究', '轴', '减', '理', '迫', '隶', '青',
          '击', '研', '厘', '书', '执', '赵', '市', '多', '己', '孪', '格', '腐', '区', '林', '导', '般', '载', '个', '骨', '三', '再', '必', '都', '解', '而', '夹', '旧', '约',
          '毫', '压', '职', '肉', '此', '到', '色', '冬', '接', '尺', '月', '输', '转', '块', '派', '矿', '社', '预', '毛', '央', '身', '显', '相', '旋', '道', '报', '拖', '洛',
          '笑', '不', '生', '恩', '识', '依', '刘', '伤', '南', '乙', '述', '场', '才', '艺', '春', '运', '向', '者', '军', '河', '织', '知', '造', '云', '扬', '印', '的', '非',
          '天', '切', '辉', '万', '升', '当', '迅', '亲', '传', '尖', '倒', '策', '固', '份', '低', '射', '病', '置', '群', '遍', '情', '打', '模', '清', '施', '移', '齿', '众',
          '口', '我', '似', '俄', '外', '节', '缺', '用', '燃', '行', '超', '么', '优', '土', '共', '透', '去', '资', '否', '脉', '要', '来', '金', '步', '松', '争', '开', '队',
          '材', '领', '感', '奴', '丰', '党', '振', '播', '矛', '出', '培', '握', '标', '平', '整', '章', '民', '缸', '杆', '给', '也', '灭', '手', '创', '急', '独', '乡', '疗',
          '容', '装', '需', '杨', '雨', '师', '由', '裂', '叶', '统', '蜂', '班', '集', '受', '扩', '滑', '些', '谣', '盾', '熟', '结', '套', '站', '却', '地', '臣', '够', '测',
          '这', '温', '塞', '克', '宜', '易', '现', '学', '帝', '端', '做', '关', '风', '绿', '义', '刷', '备', '族', '染', '福', '算', '四', '卷', '设', '阀', '阿', '复', '安',
          '鼓', '枝', '访', '比', '架', '斤', '咱', '锻', '医', '式', '收', '明', '域', '曾', '法', '孟', '达', '鱼', '献', '苏', '草', '奈', '点', '车', '阳', '旺', '后', '系',
          '面', '费', '少', '全', '担', '速', '奇', '剩', '孔', '看', '人', '判', '怕', '为', '技', '查', '持', '常', '轮', '改', '母', '鲜', '游', '雷', '吧', '今', '钢', '洋',
          '排', '肚', '退', '求', '满', '数', '努', '作', '势', '念', '药', '除', '世', '仅', '抗', '欧', '附', '心', '吹', '盘', '很', '教', '组', '近', '二', '头', '属', '目',
          '阵', '气', '许', '怎', '波', '离', '真', '直', '命', '粮', '台', '菜', '名', '板', '泥', '特', '科', '于', '侵', '密', '变', '叫', '危', '剂', '古', '每', '倾', '交',
          '幼', '与', '反', '连', '功', '电', '山', '床', '东', '助', '柱', '代', '括', '讨', '深', 'X', '展', '植', '散', '委', '选', '至', '尾', '械', '办', '察', '育', '早',
          '次', '例', '你', '号', '应', '水', '拿', '失', '伦', '对', '酸', '富', '飞', '无', '日', '表', '硬', '源', '团', '形', '样', '别', '太', '界', '溶', '斗', '远', '供',
          '险', '决', '背', '八', '阻', '零', '卒', '政', '友', '悟', '右', '村', '槽', '亳', '支', '强', '年', '合', '殖', '缘', '快', '检', '前', '保', '厂', '精', '距', '季',
          '江', '更', '突', '故', '入', '削', '害', '县', '片', '杂', '著', '主', '计', '掌', '误', '小', '力', '室', '找', '觉', '铜', '项', '封', '以', '肥', '麦', '底', '菌',
          '灌', '被', '激', '因', '该', '问', '华', '牛', '顶', '观', '务', '有', '召', '本', '校', '螺', '轻', '等', '疾', '灰', '照', '声', '部', '逮', '然', '若', '推', '影',
          '夺', '英', '文', '藏', '进', '何', '五', '钱', '拉', '原', '老', '象', '读', '见', '操', '愈', '放', '治', '经', '挥', '足', '广', '续', '壮', '武', '济', '发', '下',
          '刀', '取', '将', '重', '素', '想', '双', '获', '边', '止', '纸', '营', '根', '径', '粒', '雄', '兴', '批', '沟', '洲', '流', '如', '搞', '线', '规', '采', '动', '占',
          '参', '一', '准', '贪', '存', '胜', '球', '张', '条', '程', '越', '说', '九', '抓', '喷', '旱', '永', '史', '坚', '阴']
print(len(my_lst))

# 读取pickle文件并反序列化
# 常用汉字3755个
with open('char_dict.pkl', 'rb') as f:
    data = pickle.load(f)

# 去掉在本地测试集但是不在3375个常用字的数据6个，一共有1022个数据
for key in my_lst:
    print(key, data.get(key))