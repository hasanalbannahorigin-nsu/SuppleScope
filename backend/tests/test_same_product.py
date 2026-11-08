from app.services.research_agent.product_phase import core_product_name, same_product


def test_pack_size_form_words_and_marketing_tail_are_ignored():
    assert core_product_name("Dibedex 500 mg (30 Pcs)") == "dibedex 500"
    assert core_product_name("Dibedex 500 Capsule 60pcs Pot") == "dibedex 500"
    assert core_product_name("Healthy Plus Milk Supplement-১মাসে ৮-১০কেজি ওজন বৃদ্ধি করে") == "healthy plus milk supplement"
    assert core_product_name("Weight gain healthy plus milk supplement 120g") == "weight gain healthy plus milk supplement"


def test_listings_of_one_product_are_merged():
    # Real cases from the first live runs on the 7 ads.
    assert same_product("Dibedex 500 mg Capsule", "Dibedex 500 mg (30 Pcs)",
                        maker_a="Index Laboratories Ltd.", maker_b="Index Laboratories (Ayu)")
    assert same_product("Dibedex 500", "Dibedex 500", brand_a="Index", brand_b="Dibedex",
                        maker_a="Index laboratories (ayu) Ltd", maker_b="Index Laboratories (Ayu) Ltd")
    assert same_product("Dibedex 500", "Index Dibedex 500 Capsule",
                        maker_a="Index Laboratories (Ayu) Ltd", maker_b="ইনডেক্স ল্যাবরেটরীজ (আয়ু) লিঃ।")
    assert same_product("Super Slimming Herb", "Natural Super Slimming Herb", maker_a="JP Natural Cosmetic")
    assert same_product("Healthy Plus Milk Supplement-১মাসে ৮-১০কেজি ওজন বৃদ্ধি করে",
                        "Weight gain healthy plus milk supplement 120g")
    assert same_product("Dibedex", "Dibedex 500 mg")


def test_real_differences_keep_products_apart():
    assert not same_product("Dibedex 500 mg", "Dibedex 250 mg")
    assert not same_product("Saduri Capsule", "Saduri Plus Capsule")
    assert not same_product("Dibedex 500", "Dibedex 500", maker_a="Index Laboratories", maker_b="Square Herbal")
    assert not same_product("Centrum Silver", "Centrum Men")
    assert not same_product("Healthy Plus Milk Supplement", "Milk Supplement( Weight Gain)")
    assert not same_product("Super Slimming Herb", "Slimming Tea")
