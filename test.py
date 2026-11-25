# mod1

def mod1_input():
    filter_a = []
    filter_b = []
    pattern = []
    print("#----------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")
    for i in range(0, 3):
        if i == 0:
            print("필터 A (3줄 입력, 공백 구분)")
        elif i == 1:
            print("\n필터 B (3줄 입력, 공백 구분)")
        else:
            print("\n#----------------------------------------")
            print("# [2] 패턴 입력")
            print("#---------------------------------------")
            print("패턴 (3줄 입력, 공백 구분)")

        for j in range(0, 3):
            # 검사 해야 할 예외 : 
            # 빈 입력 [O] 
            # 열 개수 불일치 [O]
            # 문자열이 들어오는 경우 [O]

            while(1): # 예외 처리 로직
                try:
                    in_lst = list(map(float, input("").split()))
                except ValueError: # 문자열 검사 수행
                    print("숫자만 입력하세요.")
                    continue
                except KeyboardInterrupt:
                    print("Ctrl+C 입력이 감지되었습니다. 프로그램을 종료합니다.")
                    exit()
                except EOFError:
                    print("Ctrl+Z 입력이 감지되었습니다. 프로그램을 종료합니다.")
                    exit()

                if len(in_lst) != 3: # 빈 입력, 열 개수 불일치 검사 수행
                    print("행 별 숫자 3개를 입력하세요.")
                    continue
                break
            if i == 0:
                filter_a.append(in_lst)
            elif i == 1:
                filter_b.append(in_lst)
            else:
                pattern.append(in_lst)

    print(filter_a, filter_b, pattern)

def mod1():
    pass

mod1_input()

