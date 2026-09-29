from dataclasses import dataclass

from nexus.core.decision.context import DecisionContext


@dataclass(frozen=True)
class ActionPrecondition:
    """
    Describes whether a candidate action is currently executable.
    """

    valid: bool
    reason: str


class ActionPreconditions:
    """
    Validates whether NEXUS actions are executable
    in the current decision context.
    """

    @staticmethod
    def current_crop(context: DecisionContext):
        """
        Return the crop occupying the farmer's current tile.

        Returns:
            The current crop object, or None when no crop
            occupies the current tile.
        """

        farmer_x, farmer_y = context.state.farmer_position

        return next(
            (
                crop
                for crop in context.state.crops
                if crop.row == farmer_y
                and crop.col == farmer_x
            ),
            None,
        )

    def can_dig_weed(
        self,
        context: DecisionContext,
    ) -> ActionPrecondition:
        """
        Determine whether the farmer can clear a weed
        from the current tile.
        """

        farmer_x, farmer_y = context.state.farmer_position

        current_weed = next(
            (
                weed
                for weed in context.state.weeds
                if weed.col == farmer_x
                and weed.row == farmer_y
            ),
            None,
        )

        if current_weed is None:
            return ActionPrecondition(
                valid=False,
                reason="No weed exists on the current tile.",
            )

        return ActionPrecondition(
            valid=True,
            reason="The current tile contains a weed that can be cleared.",
        )

    def can_buy_seed(
        self,
        context: DecisionContext,
        crop_name: str,
    ) -> ActionPrecondition:
        """
        Determine whether a crop seed can currently be purchased.

        Seed purchasing is a resource/economic operation.

        IMPORTANT:
            Buying a seed does NOT require an immediately available
            planting tile. A purchased seed can remain in inventory
            until a suitable farm tile becomes available.

        Planting capacity is validated separately by can_plant_crop().

        Hard execution preconditions:

            1. The crop must be a known crop.
            2. The player must not already own that crop's seed.
            3. The player must have enough money.

        Existing active crops do NOT automatically prevent purchasing
        another seed of the same crop.

        The DecisionEngine is responsible for determining whether
        purchasing the seed is strategically worthwhile.
        """

        crop_name = crop_name.upper()

        # ---------------------------------------------------------
        # 1. Resolve the crop definition.
        # ---------------------------------------------------------
        crop_definition = context.crop_definition(
            crop_name
        )

        # ---------------------------------------------------------
        # 2. Prevent duplicate seed purchases.
        #
        # We only buy when there is currently no seed of this
        # crop in the player's inventory.
        # ---------------------------------------------------------
        seeds = context.seed_count(
            crop_name
        )

        if seeds > 0:
            return ActionPrecondition(
                valid=False,
                reason=(
                    f"{crop_name.title()} seed is already available."
                ),
            )

        # ---------------------------------------------------------
        # 3. Check available cash.
        # ---------------------------------------------------------
        if (
            context.state.resources.money
            < crop_definition.seed_cost
        ):
            return ActionPrecondition(
                valid=False,
                reason=(
                    f"Insufficient cash to buy "
                    f"{crop_name.lower()} seed."
                ),
            )

        # ---------------------------------------------------------
        # 4. Existing active crops do NOT invalidate the purchase.
        #
        # Example:
        #
        #   WHEAT crop is currently growing.
        #   WHEAT seed = 0.
        #   Money >= WHEAT seed cost.
        #
        # NEXUS can purchase another WHEAT seed and hold it
        # until the current crop is harvested/removed.
        # ---------------------------------------------------------
        active_crop_count = sum(
            1
            for crop in context.state.crops
            if crop.crop.upper() == crop_name
        )

        if active_crop_count > 0:
            return ActionPrecondition(
                valid=True,
                reason=(
                    f"{crop_name.title()} seed can be purchased "
                    f"for future planting; "
                    f"{active_crop_count} active "
                    f"{crop_name.lower()} crop(s) currently exist."
                ),
            )

        # ---------------------------------------------------------
        # 5. No active crop, no existing seed, and enough money.
        # ---------------------------------------------------------
        return ActionPrecondition(
            valid=True,
            reason=(
                f"{crop_name.title()} seed can be purchased "
                "with the available farm resources."
            ),
        )

    def can_buy_wheat_seed(
        self,
        context: DecisionContext,
    ) -> ActionPrecondition:
        """
        Backward-compatible wheat-specific seed purchase check.

        Wheat uses the same generic seed-purchase rules as every
        other supported crop.
        """

        return self.can_buy_seed(
            context,
            "WHEAT",
        )

    def can_plant_crop(
        self,
        context: DecisionContext,
        crop_name: str,
    ) -> ActionPrecondition:
        """
        Determine whether a specific crop can be planted
        on the farmer's current tile.
        """

        crop_name = crop_name.upper()

        # ---------------------------------------------------------
        # 1. A seed must actually exist.
        # ---------------------------------------------------------
        if context.seed_count(crop_name) <= 0:
            return ActionPrecondition(
                valid=False,
                reason=f"No {crop_name.lower()} seed is available.",
            )

        # ---------------------------------------------------------
        # 2. The current tile must not already contain a crop.
        # ---------------------------------------------------------
        current_crop = self.current_crop(
            context
        )

        if current_crop is not None:
            return ActionPrecondition(
                valid=False,
                reason="The current tile already contains a crop.",
            )

        return ActionPrecondition(
            valid=True,
            reason=(
                f"{crop_name.title()} can be planted "
                "on the current tile."
            ),
        )

    def can_plant_wheat(
        self,
        context: DecisionContext,
    ) -> ActionPrecondition:
        """
        Backward-compatible wheat-specific planting check.
        """

        return self.can_plant_crop(
            context,
            "WHEAT",
        )

    def can_water_crop(
        self,
        context: DecisionContext,
    ) -> ActionPrecondition:
        """
        Determine whether the crop on the current tile
        can be watered today.
        """

        current_crop = self.current_crop(
            context
        )

        if current_crop is None:
            return ActionPrecondition(
                valid=False,
                reason="No crop exists on the current tile.",
            )

        if current_crop.watered_today:
            return ActionPrecondition(
                valid=False,
                reason=(
                    "The current crop has already "
                    "been watered today."
                ),
            )

        return ActionPrecondition(
            valid=True,
            reason="The current crop can be watered.",
        )

    def can_harvest_crop(
        self,
        context: DecisionContext,
    ) -> ActionPrecondition:
        """
        Determine whether the crop on the current tile
        is ready to be harvested.
        """

        current_crop = self.current_crop(
            context
        )

        if current_crop is None:
            return ActionPrecondition(
                valid=False,
                reason="No crop exists on the current tile.",
            )

        if not context.is_crop_harvestable(
            current_crop
        ):
            return ActionPrecondition(
                valid=False,
                reason="The current crop is not yet harvestable.",
            )

        return ActionPrecondition(
            valid=True,
            reason="The current crop is ready for harvest.",
        )

    def can_sell_product(
        self,
        context: DecisionContext,
        product_name: str,
    ) -> ActionPrecondition:
        """
        Determine whether a product can currently be sold.

        A product is sellable only when:

            1. At least one unit exists in the farm shed.
            2. The current market price is greater than zero.

        The actual sale is executed by the agent's market-action
        layer. This method only validates the decision precondition.
        """

        product_name = product_name.upper()

        quantity = context.product_count(
            product_name
        )

        if quantity <= 0:
            return ActionPrecondition(
                valid=False,
                reason=(
                    f"No {product_name} is currently "
                    "available in the farm shed."
                ),
            )

        market_price = context.market_price(
            product_name
        )

        if market_price <= 0:
            return ActionPrecondition(
                valid=False,
                reason=(
                    f"{product_name} has no valid market "
                    "price and cannot be sold."
                ),
            )

        return ActionPrecondition(
            valid=True,
            reason=(
                f"{quantity} unit(s) of {product_name} "
                f"can be sold at {market_price:.2f} per unit."
            ),
        )