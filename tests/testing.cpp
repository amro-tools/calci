#include "catch2/matchers/catch_matchers.hpp"
#include "calci/utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <calci/defines.hpp>

TEST_CASE( "Test that always passes", "[Testing]" )
{
    
    REQUIRE( 2==2 );
}