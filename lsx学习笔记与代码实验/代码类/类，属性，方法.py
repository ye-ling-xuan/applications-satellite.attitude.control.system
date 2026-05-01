#核心关系：类（模板）→ 创建 → 对象（实例），每个对象共享类的方法，但有独立的属性值。

# 1. 定义类（类名建议用大驼峰命名法：首字母大写，无下划线）
class Person:
    # 2. 初始化方法（构造函数）：创建对象时自动执行，给对象赋初始属性
    # self：必须作为第一个参数，代表“当前创建的这个对象本身”
    def __init__(self, name, age, gender):
        # 给对象绑定属性（self.属性名 = 传入的值）
        self.name = name    # 姓名属性
        self.age = age      # 年龄属性
        self.gender = gender# 性别属性

    # 3. 普通方法：对象的“行为”，第一个参数必须是self
    def introduce(self):
        # 方法内部通过self访问对象的属性
        print(f"大家好，我叫{self.name}，今年{self.age}岁，性别{self.gender}。")
    
    #开头的 f 是 “格式化字符串” 的前缀标记，它告诉 Python：
    #这不是普通字符串，里面 {} 大括号里的内容要当成代码执行，把结果替换进字符串里。
    #运行时，{self.name} 会被替换成当前对象的 name 属性值
    #{self.age} 替换成 age 属性值，以此类推。

    #带参数的方法
    def grow_up(self, years):
        self.age += years  # 修改对象的属性值
        print(f"{self.name}长大了{years}岁，现在{self.age}岁啦！")

# 2. 创建对象（实例化）：类名(参数) → 传入__init__除self外的参数
person1 = Person("张三", 20, "男")
person2 = Person("李四", 18, "女")

# 3. 调用对象的方法
person1.introduce()  # 输出：大家好，我叫张三，今年20岁，性别男。
person2.introduce()  # 输出：大家好，我叫李四，今年18岁，性别女。

person1.grow_up(2)   # 输出：张三长大了2岁，现在22岁啦！
person2.grow_up(1)   # 输出：李四长大了1岁，现在19岁啦！

# 4. 直接访问/修改对象的属性（不推荐直接改，后面会讲封装）
print(person1.name)  # 输出：张三
person2.age = 20     # 直接修改李四的年龄
print(person2.age)   # 输出：20