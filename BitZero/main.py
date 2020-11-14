
import pickle


if __name__ == '__main__':
    with open(f"C:/development/github/puffin-trader/BitKin/IntrinsicTime/cache/order_books.pickle", 'rb') as f:
        order_books = pickle.load(f)

    print(len(order_books))
    print(order_books[0])
    print(order_books[-1])


