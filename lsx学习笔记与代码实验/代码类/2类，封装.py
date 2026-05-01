#2. 封装：隐藏内部细节（新手避坑）
#直接修改实例属性（如person.age = -10）会导致数据不合法，封装的核心是：
#用「私有属性」隐藏核心数据（属性名前加__）；
#用「getter/setter 方法」控制属性的访问和修改，增加合法性校验。
class Person:
    def __init__(self, name, age):
        self.name = name
        self.__age = age  # 私有属性：外部不能直接访问（加两个下划线）

    # getter方法：获取私有属性
    def get_age(self):
        return self.__age

    # setter方法：修改私有属性（加合法性校验）
    def set_age(self, new_age):
        if 0 < new_age < 150:  # 年龄必须在0-150之间
            self.__age = new_age
        else:
            print("年龄不合法！必须是0-150之间的数字。")

    def introduce(self):
        print(f"我叫{self.name}，今年{self.__age}岁。")

# 测试封装
p = Person("张三", 20)
p.introduce()  # 输出：我叫张三，今年20岁。

# 尝试直接访问私有属性 → 报错（封装的核心：隐藏）
# print(p.__age)  # AttributeError: 'Person' object has no attribute '__age'

# 通过getter获取年龄
print(p.get_age())  # 输出：20

# 通过setter修改年龄（合法值）
p.set_age(25)
print(p.get_age())  # 输出：25

# 尝试设置不合法年龄
p.set_age(-5)       # 输出：年龄不合法！必须是0-150之间的数字。