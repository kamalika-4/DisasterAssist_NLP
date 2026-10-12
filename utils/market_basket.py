"""
Market Basket Analysis for emergency resource requests.
Uses mlxtend's real Apriori algorithm + association_rules on synthetic
but realistic disaster-relief request transactions to discover which
relief items are commonly requested together.
"""
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from utils.data_generator import generate_market_basket_transactions


def run_market_basket_analysis(min_support=0.08, min_confidence=0.4, n_transactions=600):
    baskets = generate_market_basket_transactions(n_transactions)
    transactions = [b["items"] for b in baskets]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_enc = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_enc, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False).reset_index(drop=True)

    if frequent_itemsets.empty:
        return baskets, frequent_itemsets, pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)

    # Make itemsets human-readable strings for display
    frequent_itemsets["itemsets_str"] = frequent_itemsets["itemsets"].apply(lambda x: ", ".join(sorted(x)))
    if not rules.empty:
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    return baskets, frequent_itemsets, rules


def item_frequency_table(baskets):
    from collections import Counter
    counter = Counter()
    for b in baskets:
        counter.update(b["items"])
    df = pd.DataFrame(counter.items(), columns=["Item", "Frequency"]).sort_values("Frequency", ascending=False)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    baskets, freq, rules = run_market_basket_analysis()
    print(freq.head(10))
    print(rules.head(10)[["antecedents_str", "consequents_str", "support", "confidence", "lift"]])
