class Person:
    # 类属性：所有Person对象共享
    species = "智人"  # 人类的物种
    count = 0         # 统计创建的Person对象数量

    def __init__(self, name, age):
        # 实例属性：每个对象独立
        self.name = name
        self.age = age
        # 每次创建对象，类属性count+1
        Person.count += 1

# 创建对象
p1 = Person("张三", 20)
p2 = Person("李四", 18)

# 访问类属性（两种方式：类名.属性 / 对象.属性）
print(Person.species)  # 输出：智人
print(p1.species)      # 输出：智人（对象可访问类属性）

# 访问实例属性（只能用对象.属性）
print(p1.name)         # 输出：张三
# print(Person.name)  # 报错！类不能访问实例属性

# 类属性统计对象数量
print(Person.count)    # 输出：2（创建了2个Person对象）