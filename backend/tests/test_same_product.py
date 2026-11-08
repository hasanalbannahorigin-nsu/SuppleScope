from app.services.research_agent.product_phase import core_product_name, same_product


def test_pack_size_and_form_words_are_ignored():
    assert core_product_name("Dibedex 500 mg (30 Pcs)") == "dibedex 500"
    assert core_product_name("Dibedex 500 Capsule 60pcs Pot") == "dibedex 500"


def test_pack_variants_of_one_product_are_merged():
    # Real case from the first live run: two shop listings of the same Dibedex capsule.
    assert same_product("Dibedex 500 mg Capsule", "Dibedex 500 mg (30 Pcs)",
                        maker_a="Index Laboratories Ltd.", maker_b="Index Laboratories (Ayu)")
    assert same_product("Dibedex", "Dibedex 500 mg")


def test_real_differences_keep_products_apart():
    assert not same_product("Dibedex 500 mg", "Dibedex 250 mg")
    assert not same_product("Saduri Capsule", "Saduri Plus Capsule")
    assert not same_product("Dibedex 500", "Dibedex 500", maker_a="Index Laboratories", maker_b="Square Herbal")
    assert not same_product("Centrum Silver", "Centrum Men")
